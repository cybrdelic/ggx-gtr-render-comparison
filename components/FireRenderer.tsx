
import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import { ShaderError, ShaderParam, VideoConfig } from '../types';
import { calculateUniformLayout, writeParamsToBuffer, ParamsControlPanel } from './ShaderParams';

// --- WebGPU Type Stubs ---
type GPUDevice = any;
type GPUCanvasContext = any;
type GPURenderPipeline = any;
type GPUBuffer = any;
type GPUBindGroup = any;
declare const GPUBufferUsage: any;
declare const GPUShaderStage: any;

const getErrorMessage = (err: any): string => {
  if (err === undefined) return "Undefined Error";
  if (err === null) return "Null Error";
  if (typeof err === 'string') return err;
  if (err.reason !== undefined && err.message !== undefined) return `Device Lost (${err.reason}): ${err.message}`;
  if (err.message !== undefined) return String(err.message);
  if (err instanceof Error) return `${err.name}: ${err.message}`;
  try { const json = JSON.stringify(err); if (json !== '{}') return json; } catch (e) {}
  return String(err);
};

export interface WebGPURendererRef {
  capture: (quality?: number) => void;
  startVideo: (config: VideoConfig) => void;
  stopVideo: () => void;
  loadTexture: (file: File) => void;
  loadMesh: (file: File) => void;
  toggleAudio: () => Promise<void>;
  getComparatorValue: () => number;
}

interface WebGPURendererProps {
  shaderCode: string;
  description?: string;
  resolutionScale?: number;
  onError: (error: ShaderError) => void;
  onClearError: () => void;
  onRecordProgress: (isRecording: boolean, timeLeft: number) => void;
  onComparatorChange?: (val: number) => void;
}

const WebGPURenderer = forwardRef<WebGPURendererRef, WebGPURendererProps>(({ shaderCode, description, resolutionScale = 1.0, onError, onClearError, onRecordProgress, onComparatorChange }, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isSupported, setIsSupported] = useState<boolean>(true);
  
  const deviceRef = useRef<GPUDevice | null>(null);
  const contextRef = useRef<GPUCanvasContext | null>(null);
  const pipelineRef = useRef<GPURenderPipeline | null>(null);
  const uniformBufferRef = useRef<GPUBuffer | null>(null);
  const meshBufferRef = useRef<GPUBuffer | null>(null);
  const bvhBufferRef = useRef<GPUBuffer | null>(null);
  const bindGroupRef = useRef<GPUBindGroup | null>(null);
  const textureRef = useRef<any>(null); // Channel 0
  const samplerRef = useRef<any>(null); // Sampler
  
  const requestRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(performance.now());
  const isMountedRef = useRef<boolean>(true);
  
  // Audio State
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyzerRef = useRef<AnalyserNode | null>(null);
  const audioDataArrayRef = useRef<Uint8Array | null>(null);

  // Capture State
  const capturePendingRef = useRef<number>(0); 
  
  // Video Recording State
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recordingConfigRef = useRef<VideoConfig | null>(null);
  const recordingStartTimeRef = useRef<number>(0);
  const isRecordingRef = useRef<boolean>(false);
  const recordedFramesRef = useRef<number>(0); 
  const streamTrackRef = useRef<any>(null); 
  
  // Mesh State
  const meshTriangleCountRef = useRef<number>(0);

  // --- MATERIAL LAB PARAMS ---
  const [params, setParams] = useState<ShaderParam[]>([
    { id: 'roughness', label: 'Roughness', type: 'float', value: 0.25, min: 0.001, max: 1.0 },
    { id: 'metallic', label: 'Metallic', type: 'float', value: 0.0, min: 0.0, max: 1.0 },
    { id: 'specularF0', label: 'Specular F0', type: 'float', value: 0.5, min: 0.0, max: 1.0 },
    { id: 'comparator', label: 'Compare (L: GTR / R: GGX)', type: 'float', value: 0.5, min: 0.0, max: 1.0 },
    
    // BLACK OBSIDIAN DEFAULT TO SHOW SPECULAR
    { id: 'baseColor', label: 'Albedo', type: 'color', value: [0.02, 0.02, 0.02] }, 
    { id: 'normalScale', label: 'Surface Bump', type: 'float', value: 0.2, min: 0.0, max: 1.0 },
    
    // WORLD CLASS FEATURES
    { id: 'transmission', label: 'Glass / Transm', type: 'float', value: 0.0, min: 0.0, max: 1.0 },
    { id: 'ior', label: 'IoR (Refraction)', type: 'float', value: 1.5, min: 1.0, max: 3.0 },
    { id: 'dispersion', label: 'Spectral Disp.', type: 'float', value: 0.5, min: 0.0, max: 2.0 },
    { id: 'iridescence', label: 'Iridescence', type: 'float', value: 0.0, min: 0.0, max: 1.0 },
    { id: 'clearcoat', label: 'Clearcoat', type: 'float', value: 1.0, min: 0.0, max: 1.0 },
    
    { id: 'envIntensity', label: 'HDRI Intensity', type: 'float', value: 1.5, min: 0.0, max: 5.0 },
    
    // CAMERA PARAMS
    { id: 'focusDist', label: 'Focus Distance', type: 'float', value: 4.5, min: 1.0, max: 10.0 },
    { id: 'aperture', label: 'Aperture (Blur)', type: 'float', value: 0.2, min: 0.0, max: 2.0 },
    
    { id: 'gtrGamma', label: 'GTR Gamma (Tail)', type: 'float', value: 0.8, min: 0.5, max: 4.0 },

    // NEW FEATURES VISIBLE ON LOAD
    { id: 'volumetricDensity', label: 'Ground Fog', type: 'float', value: 0.8, min: 0.0, max: 3.0 },
    { id: 'sssStrength', label: 'SSS Strength', type: 'float', value: 0.5, min: 0.0, max: 1.0 },
    
    // MESH FX
    { id: 'twist', label: 'Space Twist', type: 'float', value: 0.0, min: -1.0, max: 1.0 },
    { id: 'wireframe', label: 'Wireframe Mix', type: 'float', value: 0.0, min: 0.0, max: 1.0 },
  ]);

  const paramsRef = useRef(params);
  useEffect(() => { 
      paramsRef.current = params; 
      const comp = params.find(p => p.id === 'comparator');
      if (comp && onComparatorChange) {
          onComparatorChange(comp.value as number);
      }
  }, [params, onComparatorChange]);

  const STANDARD_HEADER_SIZE = 48;
  const layout = calculateUniformLayout(params, STANDARD_HEADER_SIZE);
  const TOTAL_BUFFER_SIZE = 2048; // Increased for safe padding

  const cameraState = useRef({ theta: 0.5, phi: 0.3, radius: 4.5, isDragging: false, lastX: 0, lastY: 0 });
  const mouseState = useRef({ x: 0, y: 0, isDown: 0 });

  // --- HELPER: Texture Creation ---
  const createTextureFromImage = async (device: GPUDevice, source: ImageBitmap | HTMLCanvasElement) => {
    const texture = device.createTexture({
        size: [source.width, source.height, 1],
        format: 'rgba8unorm',
        usage: 0x04 | 0x02 | 0x01 | 0x10, // TEXTURE_BINDING | COPY_DST | COPY_SRC | RENDER_ATTACHMENT
    });
    device.queue.copyExternalImageToTexture(
        { source },
        { texture },
        [source.width, source.height]
    );
    return texture;
  };
  
  const createDefaultTexture = (device: GPUDevice) => {
      const size = 256;
      const canvas = document.createElement('canvas');
      canvas.width = size; canvas.height = size;
      const ctx = canvas.getContext('2d');
      if (ctx) {
          const grd = ctx.createLinearGradient(0,0,0,size);
          grd.addColorStop(0, '#1a2a6c');
          grd.addColorStop(0.5, '#b21f1f');
          grd.addColorStop(1, '#fdbb2d');
          ctx.fillStyle = grd;
          ctx.fillRect(0,0,size,size);
      }
      return createTextureFromImage(device, canvas);
  };

  // --- BVH BUILDER ---
  interface BVHNode {
      min: number[];
      max: number[];
      isLeaf: boolean;
      startIndex: number; // for leaf
      count: number; // for leaf
      leftIndex: number; // for internal
      rightIndex: number; // for internal
  }

  const buildBVH = (positions: number[][], triangleIndices: number[]) => {
      // 1. Calculate Centroids
      const centroids: number[][] = [];
      const triCount = triangleIndices.length / 3;
      for (let i = 0; i < triCount; i++) {
          const i0 = triangleIndices[i*3];
          const i1 = triangleIndices[i*3+1];
          const i2 = triangleIndices[i*3+2];
          const c = [
              (positions[i0][0] + positions[i1][0] + positions[i2][0]) / 3,
              (positions[i0][1] + positions[i1][1] + positions[i2][1]) / 3,
              (positions[i0][2] + positions[i1][2] + positions[i2][2]) / 3
          ];
          centroids.push(c);
      }

      const triIds = new Int32Array(triCount);
      for(let i=0; i<triCount; i++) triIds[i] = i;

      const finalNodes: BVHNode[] = [];
      const finalTriIndices: number[] = [];
      
      const build = (ids: Int32Array): number => {
          const nIdx = finalNodes.length;
          // Declare node strictly
          const node: BVHNode = { 
            min: [Infinity,Infinity,Infinity], 
            max: [-Infinity,-Infinity,-Infinity], 
            isLeaf: false, 
            startIndex: 0, 
            count: 0, 
            leftIndex: -1, 
            rightIndex: -1 
          };
          finalNodes.push(node);

          // Calculate BBox for this set of triangles
          for (let i = 0; i < ids.length; i++) {
              const tid = ids[i];
              for (let v = 0; v < 3; v++) {
                  const p = positions[triangleIndices[tid*3+v]];
                  node.min[0] = Math.min(node.min[0], p[0]);
                  node.min[1] = Math.min(node.min[1], p[1]);
                  node.min[2] = Math.min(node.min[2], p[2]);
                  node.max[0] = Math.max(node.max[0], p[0]);
                  node.max[1] = Math.max(node.max[1], p[1]);
                  node.max[2] = Math.max(node.max[2], p[2]);
              }
          }

          // Leaf criteria
          if (ids.length <= 4) {
              node.isLeaf = true;
              node.startIndex = finalTriIndices.length / 3;
              node.count = ids.length;
              for(let i=0; i<ids.length; i++) {
                  const tid = ids[i];
                  finalTriIndices.push(triangleIndices[tid*3], triangleIndices[tid*3+1], triangleIndices[tid*3+2]);
              }
              return nIdx;
          }

          // Split logic (Midpoint along longest axis)
          let axis = 0;
          const extent = [node.max[0]-node.min[0], node.max[1]-node.min[1], node.max[2]-node.min[2]];
          if (extent[1] > extent[0]) axis = 1;
          if (extent[2] > extent[axis]) axis = 2;

          let mid = 0;
          for(let i=0; i<ids.length; i++) mid += centroids[ids[i]][axis];
          mid /= ids.length;
          
          const leftIds: number[] = [];
          const rightIds: number[] = [];
          
          for(let i=0; i<ids.length; i++) {
              if (centroids[ids[i]][axis] < mid) leftIds.push(ids[i]); 
              else rightIds.push(ids[i]);
          }

          if (leftIds.length === 0 || rightIds.length === 0) {
               node.isLeaf = true;
               node.startIndex = finalTriIndices.length / 3;
               node.count = ids.length;
               for(let i=0; i<ids.length; i++) {
                  const tid = ids[i];
                  finalTriIndices.push(triangleIndices[tid*3], triangleIndices[tid*3+1], triangleIndices[tid*3+2]);
               }
               return nIdx;
          }
          
          node.leftIndex = build(new Int32Array(leftIds));
          node.rightIndex = build(new Int32Array(rightIds));
          return nIdx;
      };
      
      build(triIds);
      return { nodes: finalNodes, indices: finalTriIndices };
  };

  // --- HELPER: OBJ Parsing & Mesh Buffer ---
  const parseOBJ = (text: string) => {
      const positions: number[][] = [];
      const triangleIndices: number[] = [];
      const lines = text.split('\n');
      for (let line of lines) {
          line = line.trim();
          if (line.startsWith('v ')) {
              const parts = line.split(/\s+/).slice(1).map(parseFloat);
              positions.push(parts);
          } else if (line.startsWith('f ')) {
              const parts = line.split(/\s+/).slice(1);
              const vIndices = parts.map(p => parseInt(p.split('/')[0]) - 1);
              triangleIndices.push(vIndices[0], vIndices[1], vIndices[2]);
              if (vIndices.length === 4) {
                  triangleIndices.push(vIndices[0], vIndices[2], vIndices[3]);
              }
          }
      }
      return { positions, triangleIndices };
  };

  const createMeshBuffers = (device: GPUDevice, positions: number[][], indices: number[]) => {
      // Pack Triangles (Re-ordered by BVH if applied)
      const data = new Float32Array((indices.length / 3) * 12);
      let offset = 0;
      for (let i = 0; i < indices.length; i += 3) {
          const v0 = positions[indices[i]];
          const v1 = positions[indices[i+1]];
          const v2 = positions[indices[i+2]];
          data[offset++] = v0[0]; data[offset++] = v0[1]; data[offset++] = v0[2]; data[offset++] = 0;
          data[offset++] = v1[0]; data[offset++] = v1[1]; data[offset++] = v1[2]; data[offset++] = 0;
          data[offset++] = v2[0]; data[offset++] = v2[1]; data[offset++] = v2[2]; data[offset++] = 0;
      }

      // Buffer
      const meshBuffer = device.createBuffer({
          size: Math.max(data.byteLength, 1024),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(meshBuffer, 0, data);
      
      return { meshBuffer, count: indices.length / 3 };
  };

  const createBVHBuffer = (device: GPUDevice, nodes: BVHNode[]) => {
      // Node Size: 32 bytes (vec3 min, f32 data1, vec3 max, f32 data2)
      const data = new Float32Array(nodes.length * 8);
      for(let i=0; i<nodes.length; i++) {
          const n = nodes[i];
          const off = i * 8;
          data[off+0] = n.min[0]; data[off+1] = n.min[1]; data[off+2] = n.min[2];
          
          if (n.isLeaf) {
              // leaf: data1 = -startIndex - 1
              data[off+3] = -n.startIndex - 1;
              // data2 = count
              data[off+7] = n.count;
          } else {
              // internal: data1 = left, data2 = right
              data[off+3] = n.leftIndex;
              data[off+7] = n.rightIndex;
          }
          data[off+4] = n.max[0]; data[off+5] = n.max[1]; data[off+6] = n.max[2];
      }
      
      const buffer = device.createBuffer({
          size: Math.max(data.byteLength, 1024),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
  };

  useImperativeHandle(ref, () => ({
    capture: (quality = 1) => {
      capturePendingRef.current = quality;
    },
    loadTexture: async (file: File) => {
        if (!deviceRef.current || !file) return;
        try {
            const bitmap = await createImageBitmap(file);
            const texture = await createTextureFromImage(deviceRef.current, bitmap);
            textureRef.current = texture;
            rebind(deviceRef.current);
        } catch (e) {
            console.error("Failed to load texture", e);
        }
    },
    loadMesh: async (file: File) => {
        if (!deviceRef.current || !file) return;
        const reader = new FileReader();
        reader.onload = async (e) => {
            const text = e.target?.result as string;
            const { positions, triangleIndices } = parseOBJ(text);
            
            // Build BVH
            const { nodes, indices: sortedIndices } = buildBVH(positions, triangleIndices);
            
            // Create Buffers
            if (meshBufferRef.current) meshBufferRef.current.destroy();
            const { meshBuffer, count } = createMeshBuffers(deviceRef.current!, positions, sortedIndices);
            meshBufferRef.current = meshBuffer;
            meshTriangleCountRef.current = count;
            
            if (bvhBufferRef.current) bvhBufferRef.current.destroy();
            const bvhBuffer = createBVHBuffer(deviceRef.current!, nodes);
            bvhBufferRef.current = bvhBuffer;
            
            rebind(deviceRef.current!);
        };
        reader.readAsText(file);
    },
    toggleAudio: async () => {
        if (audioContextRef.current) {
            audioContextRef.current.suspend();
            audioContextRef.current = null;
            return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const ctx = new AudioContext();
            const source = ctx.createMediaStreamSource(stream);
            const analyzer = ctx.createAnalyser();
            analyzer.fftSize = 256;
            source.connect(analyzer);
            
            audioContextRef.current = ctx;
            analyzerRef.current = analyzer;
            audioDataArrayRef.current = new Uint8Array(analyzer.frequencyBinCount);
        } catch (e) {
            console.error("Audio init failed", e);
            alert("Could not access microphone.");
        }
    },
    startVideo: (config: VideoConfig) => {
        if (!canvasRef.current) return;
        recordingConfigRef.current = config;
        chunksRef.current = [];
        recordedFramesRef.current = 0;
        canvasRef.current.width = 1920;
        canvasRef.current.height = 1080;

        const stream = canvasRef.current.captureStream(0);
        const track = stream.getVideoTracks()[0];
        let targetStream = stream;

        if (track && (track as any).requestFrame) {
             streamTrackRef.current = track;
        } else {
             // Fallback for browsers without requestFrame support for CanvasCapture
             streamTrackRef.current = null;
             targetStream = canvasRef.current.captureStream(config.fps);
        }

        let mimeType = 'video/webm;codecs=vp9';
        if (typeof MediaRecorder !== 'undefined' && !MediaRecorder.isTypeSupported(mimeType)) {
             mimeType = 'video/webm;codecs=vp8';
        }

        if (!recorderRef.current) {
            recorderRef.current = new MediaRecorder(targetStream, {
                mimeType,
                videoBitsPerSecond: config.bitrate * 1000000
            });
        }

        const activeRecorder = recorderRef.current!;
        activeRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunksRef.current.push(e.data);
        };
        activeRecorder.onstop = () => {
            const blob = new Blob(chunksRef.current, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `material_lab_${Date.now()}.webm`;
            a.click();
            URL.revokeObjectURL(url);
            isRecordingRef.current = false;
            streamTrackRef.current = null;
            onRecordProgress(false, 0);
        };
        activeRecorder.start();
        recordingStartTimeRef.current = performance.now();
        isRecordingRef.current = true;
    },
    stopVideo: () => {
        if (recorderRef.current && recorderRef.current.state === 'recording') {
            recorderRef.current.stop();
        }
    },
    getComparatorValue: () => {
        const param = paramsRef.current.find(p => p.id === 'comparator');
        return (param?.value as number) || 0.5;
    }
  }));

  const rebind = (device: GPUDevice) => {
      if (!pipelineRef.current || !uniformBufferRef.current || !textureRef.current || !samplerRef.current || !meshBufferRef.current || !bvhBufferRef.current) return;
      
      const bindGroup = device.createBindGroup({
          layout: pipelineRef.current.getBindGroupLayout(0),
          entries: [
              { binding: 0, resource: { buffer: uniformBufferRef.current } },
              { binding: 1, resource: textureRef.current.createView() },
              { binding: 2, resource: samplerRef.current },
              { binding: 3, resource: { buffer: meshBufferRef.current } },
              { binding: 4, resource: { buffer: bvhBufferRef.current } }
          ]
      });
      bindGroupRef.current = bindGroup;
  };

  const compilePipeline = async (device: GPUDevice, code: string, context: GPUCanvasContext) => {
      const format = (navigator as any).gpu.getPreferredCanvasFormat();
      
      const shaderModule = device.createShaderModule({ label: 'Main', code });
      const compilationInfo = await shaderModule.getCompilationInfo();
      if (compilationInfo.messages.length > 0) {
        let hasError = false;
        for (const msg of compilationInfo.messages) {
          if (msg.type === 'error') {
              hasError = true;
              onError({ type: 'compilation', message: getErrorMessage(msg.message), lineNum: msg.lineNum, linePos: msg.linePos });
          }
        }
        if (hasError) return;
      }
      onClearError();

      const bindGroupLayout = device.createBindGroupLayout({ 
          entries: [
              { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' }},
              { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
              { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
              { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
              { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
          ]
      });

      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
      const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: { module: shaderModule, entryPoint: 'vs_main' },
        fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      });
      pipelineRef.current = pipeline;
      rebind(device);
  };

  useEffect(() => {
    isMountedRef.current = true;
    const initWebGPU = async () => {
      const gpu = (navigator as any).gpu;
      if (!gpu) { setIsSupported(false); onError({ type: 'compilation', message: "WebGPU not supported." }); return; }

      try {
        const adapter = await gpu.requestAdapter();
        if (!adapter) { setIsSupported(false); onError({ type: 'compilation', message: "No GPU adapter." }); return; }
        const device = await adapter.requestDevice();
        if (!isMountedRef.current) { device.destroy(); return; }
        deviceRef.current = device;

        device.lost.then((info: any) => { if (isMountedRef.current) onError({ type: 'runtime', message: getErrorMessage(info) }); });
        device.addEventListener('uncapturederror', (e: any) => { if (isMountedRef.current) onError({ type: 'runtime', message: getErrorMessage(e.error) }); });

        const canvas = canvasRef.current;
        if (!canvas) return;
        const context = canvas.getContext('webgpu') as any;
        contextRef.current = context;
        const format = gpu.getPreferredCanvasFormat();
        context.configure({ device, format, alphaMode: 'opaque' });

        const uniformBuffer = device.createBuffer({ size: TOTAL_BUFFER_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        uniformBufferRef.current = uniformBuffer;
        
        // DEFAULT MESH (Cube) + BVH
        const cubeObj = `v -0.5 -0.5 -0.5\nv 0.5 -0.5 -0.5\nv 0.5 0.5 -0.5\nv -0.5 0.5 -0.5\nv -0.5 -0.5 0.5\nv 0.5 -0.5 0.5\nv 0.5 0.5 0.5\nv -0.5 0.5 0.5\nf 1 2 3\nf 1 3 4\nf 5 6 7\nf 5 7 8\nf 1 5 8\nf 1 8 4\nf 2 6 7\nf 2 7 3\nf 1 5 6\nf 1 6 2\nf 4 8 7\nf 4 7 3`;
        const { positions, triangleIndices } = parseOBJ(cubeObj);
        const { nodes, indices } = buildBVH(positions, triangleIndices);
        const { meshBuffer, count } = createMeshBuffers(device, positions, indices);
        meshBufferRef.current = meshBuffer;
        meshTriangleCountRef.current = count;
        bvhBufferRef.current = createBVHBuffer(device, nodes);

        const defaultTex = await createDefaultTexture(device);
        textureRef.current = defaultTex;
        const sampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        });
        samplerRef.current = sampler;

        await compilePipeline(device, shaderCode, context);
        requestRef.current = requestAnimationFrame(render);
      } catch (err: any) { onError({ type: 'compilation', message: getErrorMessage(err) }); }
    };
    initWebGPU();
    return () => { isMountedRef.current = false; if (requestRef.current !== null) cancelAnimationFrame(requestRef.current); };
  }, []);

  useEffect(() => {
      if (deviceRef.current && contextRef.current) {
          compilePipeline(deviceRef.current, shaderCode, contextRef.current);
      }
  }, [shaderCode]);

  const render = (time: number) => {
    const device = deviceRef.current;
    const context = contextRef.current;
    const pipeline = pipelineRef.current;
    const uniformBuffer = uniformBufferRef.current;
    const bindGroup = bindGroupRef.current;
    const canvas = canvasRef.current;

    if (!device || !context || !pipeline || !uniformBuffer || !bindGroup || !canvas) {
         requestRef.current = requestAnimationFrame(render);
         return;
    }

    let width, height;
    if (capturePendingRef.current > 0) {
        width = 3840; height = 2160;
        canvas.width = width; canvas.height = height;
    } else if (isRecordingRef.current) {
        width = 1920; height = 1080;
        if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    } else {
        const dpr = (window.devicePixelRatio || 1) * resolutionScale; 
        width = Math.floor(canvas.clientWidth * dpr);
        height = Math.floor(canvas.clientHeight * dpr);
        if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    }

    if (width === 0 || height === 0) {
        requestRef.current = requestAnimationFrame(render);
        return;
    }

    let elapsedTime = (time - startTimeRef.current) * 0.001;
    let cameraTheta = cameraState.current.theta;
    let cameraPhi = cameraState.current.phi;
    let cameraRadius = cameraState.current.radius;
    
    const currentParams = [...paramsRef.current];

    if (isRecordingRef.current && recordingConfigRef.current) {
        const fps = recordingConfigRef.current.fps;
        elapsedTime = recordedFramesRef.current / fps;
        recordedFramesRef.current++;
        const duration = recordingConfigRef.current.duration;
        onRecordProgress(true, duration - elapsedTime);

        const shot = recordingConfigRef.current.shotType;
        if (shot === 'orbit') {
            cameraTheta += elapsedTime * 0.5;
        } else if (shot === 'sweep') {
            cameraTheta += elapsedTime * 0.3; cameraPhi = 0.1; cameraRadius = 6.0;
        }

        if (recordingConfigRef.current.orchestrate) {
             const roughIndex = currentParams.findIndex(p => p.id === 'roughness');
             if (roughIndex !== -1) {
                 const p = { ...currentParams[roughIndex] } as any; p.value = 0.5 + Math.sin(elapsedTime) * 0.4; currentParams[roughIndex] = p;
             }
        }

        if (elapsedTime >= duration) {
             if (recorderRef.current && recorderRef.current.state === 'recording') recorderRef.current.stop();
        }
    }

    const cx = cameraRadius * Math.cos(cameraPhi) * Math.sin(cameraTheta);
    const cy = cameraRadius * Math.sin(cameraPhi);
    const cz = cameraRadius * Math.cos(cameraPhi) * Math.cos(cameraTheta);
    
    const uniformData = new Float32Array(TOTAL_BUFFER_SIZE / 4); 
    uniformData[0] = width; uniformData[1] = height; uniformData[2] = elapsedTime;
    uniformData[4] = cx; uniformData[5] = cy; uniformData[6] = cz;
    uniformData[8] = mouseState.current.x; uniformData[9] = mouseState.current.y; uniformData[10] = mouseState.current.isDown;
    writeParamsToBuffer(uniformData, currentParams, layout);

    // MANUAL WRITE INDICES
    // Params end at 132 bytes (wireframe)
    // Mesh Count -> Offset 136 -> Index 34
    uniformData[34] = meshTriangleCountRef.current;
    
    // Render Mode -> Offset 140 -> Index 35
    if (capturePendingRef.current > 0) {
        uniformData[35] = capturePendingRef.current; 
    } else if (isRecordingRef.current) {
        uniformData[35] = 2.0;
    } else {
        uniformData[35] = 0.0;
    }

    // Audio -> Offset 144 -> Index 36
    let low = 0, mid = 0, high = 0, vol = 0;
    if (analyzerRef.current && audioDataArrayRef.current) {
        analyzerRef.current.getByteFrequencyData(audioDataArrayRef.current);
        const data = audioDataArrayRef.current;
        const bufferLength = data.length;
        vol = data.reduce((a,b)=>a+b, 0) / bufferLength / 255.0;
    }
    uniformData[36] = low; uniformData[37] = mid; uniformData[38] = high; uniformData[39] = vol;
    
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();
    const renderPass = commandEncoder.beginRenderPass({ colorAttachments: [{ view: textureView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    renderPass.setPipeline(pipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(6);
    renderPass.end();
    device.queue.submit([commandEncoder.finish()]);

    if (isRecordingRef.current && streamTrackRef.current && (streamTrackRef.current as any).requestFrame) {
        (streamTrackRef.current as any).requestFrame();
    }

    if (capturePendingRef.current > 0) {
        const link = document.createElement('a');
        link.download = `material_lab_${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png', 1.0);
        link.click();
        capturePendingRef.current = 0;
    }

    requestRef.current = requestAnimationFrame(render);
  };

  const handlePointerDown = (e: React.PointerEvent) => { 
      if (isRecordingRef.current) return; 
      canvasRef.current?.setPointerCapture(e.pointerId); 
      cameraState.current.isDragging = true; 
      cameraState.current.lastX = e.clientX; 
      cameraState.current.lastY = e.clientY; 
      mouseState.current.isDown = 1.0; 
  };
  const handlePointerMove = (e: React.PointerEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if(rect) { mouseState.current.x = e.clientX - rect.left; mouseState.current.y = e.clientY - rect.top; }
    if (cameraState.current.isDragging) {
      const dx = e.clientX - cameraState.current.lastX; const dy = e.clientY - cameraState.current.lastY;
      cameraState.current.lastX = e.clientX; cameraState.current.lastY = e.clientY;
      cameraState.current.theta -= dx * 0.01; cameraState.current.phi += dy * 0.01;
      cameraState.current.phi = Math.max(-1.5, Math.min(1.5, cameraState.current.phi));
    }
  };
  const handlePointerUp = (e: React.PointerEvent) => { canvasRef.current?.releasePointerCapture(e.pointerId); cameraState.current.isDragging = false; mouseState.current.isDown = 0.0; };
  const handleWheel = (e: React.WheelEvent) => { cameraState.current.radius = Math.max(1.5, Math.min(20.0, cameraState.current.radius + e.deltaY * 0.005)); };

  if (!isSupported) return <div className="w-full h-full flex items-center justify-center bg-black text-red-500 font-mono"><p>WebGPU not supported.</p></div>;

  return (
    <>
        <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair touch-none" onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={handlePointerUp} onPointerLeave={handlePointerUp} onWheel={handleWheel} />
        <ParamsControlPanel params={params} setParams={setParams} description={description} />
    </>
  );
});

export default WebGPURenderer;
