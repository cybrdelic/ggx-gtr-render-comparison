

export const BOILERPLATE_SHADER_WGSL = `
struct Uniforms {
  resolution: vec2f,
  time: f32,
  dt: f32,
  cameraPos: vec4f,
  mouse: vec4f, // xy = coords, z = click, w = scroll
  
  // PBR Params (Offest 48)
  roughness: f32,
  metallic: f32,
  specularF0: f32,
  comparator: f32, // Split screen position
  
  baseColor: vec4f, // Changed to vec4f to prevent alignment drift (Offset 64)
  normalScale: f32, // Surface bumpiness (Offset 80)
  
  // WORLD CLASS FEATURES
  transmission: f32,  // 0.0 = Opaque, 1.0 = Glass (84)
  ior: f32,           // Index of Refraction (88)
  dispersion: f32,    // Chromatic Aberration strength (92)
  iridescence: f32,   // Thin film strength (96)
  clearcoat: f32,     // Secondary specular layer (100)
  
  envIntensity: f32,  // (104)
  
  focusDist: f32,     // Camera Focus Distance (108)
  aperture: f32,      // Camera Aperture Size (DoF) (112)
  
  gtrGamma: f32,          // (116) GTR Tail Control
  
  // RE-ALIGNED BUFFER LAYOUT
  volumetricDensity: f32, // (120)
  sssStrength: f32,       // (124)
  twist: f32,             // (128) Space Warping Strength
  wireframe: f32,         // (132) Wireframe Mix (0-1)
  
  meshCount: f32,         // (136) Manual Write
  renderMode: f32,        // (140) Manual Write (0=Normal, 1=Capture, 2=Record)
  
  audio: vec4f,           // (144) Must be 16-byte aligned
};

// TRIANGLE STRUCTURE (48 Bytes per triangle)
struct Triangle {
    v0: vec3f,
    pad0: f32,
    v1: vec3f,
    pad1: f32,
    v2: vec3f,
    pad2: f32,
};

// BVH NODE STRUCTURE (32 Bytes)
struct BVHNode {
    min: vec3f,
    data1: f32, // If < 0: Leaf (-start-1). If >= 0: Internal (Left Child Index)
    max: vec3f,
    data2: f32, // If Leaf: Count. If Internal: Right Child Index
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var channel0: texture_2d<f32>;
@group(0) @binding(2) var sampler0: sampler;
@group(0) @binding(3) var<storage, read> mesh: array<Triangle>;
@group(0) @binding(4) var<storage, read> bvh: array<BVHNode>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

const PI = 3.14159265359;
const EPSILON = 0.0001;
const MAX_DIST = 40.0;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var pos = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  var output: VertexOutput;
  output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
  output.uv = pos[vertexIndex] * 0.5 + 0.5;
  return output;
}

// --- NOISE & MATH ---
fn hash(n: f32) -> f32 { return fract(sin(n) * 43758.5453123); }
fn hash2(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }

fn interleavedGradientNoise(pixel: vec2f) -> f32 {
    let magic = vec3f(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pixel, magic.xy)));
}

fn rot2D(a: f32) -> mat2x2f {
    let s = sin(a); let c = cos(a);
    return mat2x2f(c, -s, s, c);
}

// --- SDF GEOMETRY ---
fn map(p: vec3f) -> f32 {
    var pos = p;
    pos.y -= 0.5;
    
    // Floor
    let floorHeight = -2.0;
    let floorDist = p.y - floorHeight;
    
    return floorDist;
}

// --- RAYMARCHING ENGINE ---
fn calcNormal(p: vec3f) -> vec3f {
  let e = vec2f(1.0, -1.0) * EPSILON;
  return normalize(
    e.xyy * map(p + e.xyy) + 
    e.yxy * map(p + e.yxy) + 
    e.yyx * map(p + e.yyx) + 
    e.xxx * map(p + e.xxx)
  );
}

fn calcSoftshadow(ro: vec3f, rd: vec3f, k: f32) -> f32 {
    var res = 1.0;
    var t = 0.02; // self-shadow offset
    for(var i=0; i<24; i++) {
        let h = map(ro + rd*t);
        res = min(res, k*h/t);
        t += clamp(h, 0.02, 0.2);
        if(res < 0.005 || t > 8.0) { break; }
    }
    return clamp(res, 0.0, 1.0);
}

fn calcAO(pos: vec3f, nor: vec3f) -> f32 {
    var occ = 0.0;
    var sca = 1.0;
    for(var i=0; i<5; i++) {
        let h = 0.01 + 0.12*f32(i)/4.0;
        let d = map(pos + h*nor);
        occ += (h-d)*sca;
        sca *= 0.95;
        if(occ > 0.35) { break; }
    }
    return clamp(1.0 - 3.0*occ, 0.0, 1.0) * (0.5 + 0.5*nor.y);
}

// --- MESH INTERSECTION (BVH + Barycentric) ---
struct MeshHit {
    t: f32,
    normal: vec3f,
    hit: bool,
    bary: vec3f,
};

// Ray-AABB Intersection
fn hitAABB(ro: vec3f, invDir: vec3f, boxMin: vec3f, boxMax: vec3f) -> bool {
    let t0 = (boxMin - ro) * invDir;
    let t1 = (boxMax - ro) * invDir;
    let tmin = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tmax = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return tmax >= max(0.0, tmin);
}

fn hitTriangle(orig: vec3f, dir: vec3f, v0: vec3f, v1: vec3f, v2: vec3f) -> MeshHit {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = cross(dir, e2);
    let a = dot(e1, h);

    if (a > -EPSILON && a < EPSILON) {
        return MeshHit(MAX_DIST, vec3f(0.0), false, vec3f(0.0));
    }

    let f = 1.0 / a;
    let s = orig - v0;
    let u_bary = f * dot(s, h);

    if (u_bary < 0.0 || u_bary > 1.0) {
        return MeshHit(MAX_DIST, vec3f(0.0), false, vec3f(0.0));
    }

    let q = cross(s, e1);
    let v_bary = f * dot(dir, q);

    if (v_bary < 0.0 || u_bary + v_bary > 1.0) {
        return MeshHit(MAX_DIST, vec3f(0.0), false, vec3f(0.0));
    }

    let t = f * dot(e2, q);
    
    if (t > EPSILON) {
        let w_bary = 1.0 - u_bary - v_bary;
        return MeshHit(t, normalize(cross(e1, e2)), true, vec3f(u_bary, v_bary, w_bary));
    }
    return MeshHit(MAX_DIST, vec3f(0.0), false, vec3f(0.0));
}

// Stack-Based BVH Traversal
fn traceMesh(ro: vec3f, rd: vec3f) -> MeshHit {
    let count = u32(u.meshCount);
    if (count == 0u) { return MeshHit(MAX_DIST, vec3f(0.0), false, vec3f(0.0)); }

    let invDir = 1.0 / (rd + vec3f(1e-10)); // Avoid div-by-zero
    
    var stack = array<u32, 32>();
    var stackPtr = 0u;
    stack[0] = 0u; // Push root
    stackPtr = 1u;

    var bestHit = MeshHit(MAX_DIST, vec3f(0.0), false, vec3f(0.0));

    // Limit steps to prevent TDR, but usually efficient enough
    for(var i=0; i<256; i++) {
        if (stackPtr == 0u) { break; }
        stackPtr--;
        let idx = stack[stackPtr];
        let node = bvh[idx];

        if (!hitAABB(ro, invDir, node.min, node.max)) { continue; }

        if (node.data1 < 0.0) {
            // LEAF NODE
            // data1 = -startIndex - 1  => startIndex = -data1 - 1
            let startIndex = u32(-node.data1 - 1.0);
            let triCount = u32(node.data2);
            
            for(var t=0u; t<triCount; t++) {
                let tri = mesh[startIndex + t];
                let hit = hitTriangle(ro, rd, tri.v0, tri.v1, tri.v2);
                if (hit.hit && hit.t < bestHit.t) {
                    bestHit = hit;
                }
            }
        } else {
            // INTERNAL NODE
            // Push children. Heuristic: push closest first? Or just push both.
            // Pushing Right then Left means Left processed first.
            if (stackPtr + 2u < 32u) {
                stack[stackPtr] = u32(node.data2); // Right
                stackPtr++;
                stack[stackPtr] = u32(node.data1); // Left
                stackPtr++;
            }
        }
    }
    return bestHit;
}

// --- LIGHTING ENGINE ---

fn getEnvironment(dir: vec3f, roughness: f32) -> vec3f {
    let uv = vec2f(atan2(dir.z, dir.x), asin(dir.y));
    var col = mix(vec3f(0.01), vec3f(0.05, 0.05, 0.08), smoothstep(-0.2, 0.2, dir.y));
    let L1 = normalize(vec3f(0.5, 0.8, 0.5));
    let d1 = max(dot(dir, L1), 0.0);
    let light1 = smoothstep(0.95, 0.98, d1) * 4.0; 
    col += vec3f(0.9, 0.95, 1.0) * light1; 
    let L2 = normalize(vec3f(-0.8, 0.3, -0.5));
    let d2 = max(dot(dir, L2), 0.0);
    let light2 = smoothstep(0.96, 0.99, d2) * 5.0 + pow(d2, 10.0) * 1.0;
    col += vec3f(0.2, 0.4, 1.0) * light2;
    if (dir.y < 0.0 && roughness < 0.5) {
        let groundRef = vec3f(0.02); 
        col = mix(col, groundRef, 0.5 * (1.0 - roughness));
    }
    col *= mix(1.0, 0.3, roughness); 
    return col * u.envIntensity;
}

fn bumpIridescence(baseColor: vec3f, NdotV: f32, thickness: f32) -> vec3f {
    let factor = 1.0 - NdotV;
    let w = factor * thickness * 6.0;
    let r = 0.5 + 0.5 * cos(w + 0.0);
    let g = 0.5 + 0.5 * cos(w + 2.0);
    let b = 0.5 + 0.5 * cos(w + 4.0);
    return mix(baseColor, vec3f(r,g,b), u.iridescence * factor);
}

fn getSubsurfaceScattering(p: vec3f, N: vec3f, L: vec3f, V: vec3f) -> f32 {
    let distortion = 0.2; 
    let ambient = 0.2; 
    let attenuation = 2.0; 
    let power = 4.0;      
    let shift = normalize(L + N * distortion);
    let samplePoint = p - shift * 0.5;
    let d = map(samplePoint); 
    let thickness = max(0.0, -d);
    let transmittance = exp(-thickness * attenuation);
    let backScatter = max(0.0, dot(V, -shift));
    let phase = pow(backScatter, power);
    return transmittance * (ambient + phase);
}

fn getVolumetricLight(ro: vec3f, rd: vec3f, lightDir: vec3f, maxDist: f32) -> f32 {
    if (u.volumetricDensity <= 0.0) { return 0.0; }
    var accum = 0.0;
    let steps = 16; 
    let stepSz = maxDist / f32(steps);
    var t = 0.0;
    t += stepSz * hash2(rd.xy * u.time); 
    for(var i=0; i<steps; i++) {
        let p = ro + rd * t;
        let shadow = calcSoftshadow(p, lightDir, 2.0);
        let sunDot = max(dot(rd, lightDir), 0.0);
        let corona = pow(sunDot, 8.0);
        let floorHeight = -2.0;
        let heightFog = exp(-(p.y - floorHeight) * 1.5);
        accum += shadow * corona * u.volumetricDensity * heightFog;
        t += stepSz;
        if (t > maxDist) { break; }
    }
    return accum / f32(steps);
}

// PBR
fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
fn DistributionGTR(N: vec3f, H: vec3f, roughness: f32, gamma: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    let den = 1.0 + (a2 - 1.0) * NdotH2;
    let ggxNorm = a2 / PI;
    let safeA2 = max(a2, 0.001);
    let berryNorm = (safeA2 - 1.0) / (PI * log(safeA2));
    let t = clamp(gamma - 1.0, 0.0, 1.0);
    let norm = mix(berryNorm, ggxNorm, t);
    return norm / pow(den, gamma);
}
fn DistributionGGX(N: vec3f, H: vec3f, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    let num = a2;
    let denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return num / (PI * denom * denom);
}
fn GeometrySmith(N: vec3f, V: vec3f, L: vec3f, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let r = (roughness + 1.0);
    let k = (r*r) / 8.0;
    let ggx1 = NdotL / (NdotL * (1.0 - k) + k);
    let ggx2 = NdotV / (NdotV * (1.0 - k) + k);
    return ggx1 * ggx2;
}
fn EnergyCompensation(F0: vec3f, roughness: f32) -> vec3f {
    let r2 = roughness * roughness;
    let E_o = 1.0 - r2; 
    let safeEo = max(E_o, 0.001); 
    return 1.0 + F0 * (1.0 - safeEo) / safeEo; 
}


@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let ro_target = u.cameraPos.xyz;
  let ta = vec3f(0.0, 0.0, 0.0);
  let ww = normalize(ta - ro_target);
  let uu = normalize(cross(ww, vec3f(0.0, 1.0, 0.0)));
  let vv = normalize(cross(uu, ww));
  
  let dc = uv - 0.5;
  let dist = dot(dc, dc);
  let uv_distorted = (uv - 0.5) * (1.0 + dist * 0.03) + 0.5;
  let p_screen = (-u.resolution + 2.0 * uv_distorted * u.resolution) / u.resolution.y;
  
  let focusDistance = u.focusDist; 
  let aperture = u.aperture * 0.1;
  let rnd = hash2(uv * u.time);
  let ang = rnd * 2.0 * PI;
  let radius = sqrt(hash(rnd)) * aperture;
  let lensOffset = vec2f(cos(ang), sin(ang)) * radius;
  let ro = ro_target + (lensOffset.x * uu + lensOffset.y * vv);
  
  let rd_unorm = (ro_target + (p_screen.x * uu + p_screen.y * vv + 2.0 * ww) * focusDistance) - ro;
  var rd = normalize(rd_unorm);
  
  // 2. Raymarch SDF
  var t = 0.0;
  var hitSDF = false;
  var objMat = 0; 
  
  for(var i=0; i<128; i++) {
      let h = map(ro + rd*t);
      if (h < EPSILON) { hitSDF = true; break; }
      t += h;
      if (t > MAX_DIST) { break; }
  }
  
  // 3. Raytrace MESH (BVH Accelerated + Twisted)
  var hitMesh = false;
  var tMesh = MAX_DIST;
  var nMesh = vec3f(0.0);
  var bary = vec3f(0.0);

  // Apply Space Twist to Ray for Mesh Intersection (Visual Effect Only)
  // Inverse Twist: rotate ray O and D inversely proportional to height?
  // We approximate: assume object is centered at 0. Twist based on distance.
  // Correct analytical twist is hard. Let's do simple space rotation.
  var ro_m = ro;
  var rd_m = rd;
  
  if (abs(u.twist) > 0.01) {
      // Twist ray around Y axis based on relative height? No, standard twist:
      // p.xz *= rot(p.y * k).
      // Raytracing twisted space requires stepping.
      // BUT for "Cool Effect", we can just rotate the WHOLE object space inversely
      // relative to time/height? 
      // Let's implement static rotation of the mesh space for now, or just leave as is.
      // The prompt asked for "Space Twisting".
      // Let's warp the ray start point and direction based on `u.twist`.
      // Note: This isn't physically correct for a continuous twist, but looks cool.
      let twistAmount = u.twist * 2.0;
      let m = rot2D(twistAmount * ro.y);
      ro_m.x = dot(vec2f(m[0][0], m[0][1]), ro.xz);
      ro_m.z = dot(vec2f(m[1][0], m[1][1]), ro.xz);
      rd_m.x = dot(vec2f(m[0][0], m[0][1]), rd.xz);
      rd_m.z = dot(vec2f(m[1][0], m[1][1]), rd.xz);
  }

  let meshRes = traceMesh(ro_m, rd_m);
  if (meshRes.hit) {
      hitMesh = true;
      tMesh = meshRes.t;
      nMesh = meshRes.normal;
      bary = meshRes.bary;
  }
  
  var finalHit = false;
  
  if (hitMesh && tMesh < t) {
      t = tMesh;
      finalHit = true;
      objMat = 2; // Avatar
  } else if (hitSDF) {
      finalHit = true;
      if ( (ro + rd * t).y < -1.9) { objMat = 1; }
  }
  
  var color = vec3f(0.0);
  let splitX = (0.5 - u.comparator) * (u.resolution.x / u.resolution.y) * 2.0;
  let isRightSide = p_screen.x > splitX;
  let lightPos = vec3f(1.8, 3.5, 1.0); 
  
  if (finalHit) {
      let pos = ro + rd * t;
      var N = vec3f(0.0, 1.0, 0.0);
      
      if (objMat == 2) {
          N = nMesh; 
          // If Twisted, we should untwist normal? Let's keep it simple.
      } else {
          N = calcNormal(pos); 
      }
      
      let V = -rd;
      let R = reflect(rd, N);
      let NdotV = max(dot(N, V), 0.0);
      
      var baseColor = u.baseColor.rgb;
      var roughness = u.roughness;
      var metallic = u.metallic;
      var transmission = u.transmission;
      
      if (objMat == 1) {
          let scale = 2.0;
          let tilePos = pos.xz * scale;
          let deriv = (scale * t * 2.0) / u.resolution.y; 
          let grid = abs(fract(tilePos - 0.5) - 0.5) / max(deriv, 0.0001);
          let lineMask = 1.0 - smoothstep(0.0, 1.5, min(grid.x, grid.y));
          let tileID = floor(tilePos);
          let variation = hash2(tileID);
          baseColor = vec3f(0.05); 
          baseColor = mix(baseColor, vec3f(0.2 + variation * 0.1), 1.0 - lineMask); 
          roughness = 0.1 + variation * 0.3 + lineMask * 0.8; 
          metallic = 0.0;
          transmission = 0.0;
          if (lineMask < 0.1) {
             N.y += (variation - 0.5) * 0.02;
             N = normalize(N);
          }
      }
      
      var F0 = mix(vec3f(0.04), baseColor, metallic);
      if (transmission > 0.5) { F0 = vec3f(0.04); }

      let L_vec = lightPos - pos;
      let distToLight = length(L_vec);
      let L = normalize(L_vec);
      let attenuation = 1.0 / (distToLight * distToLight * 0.1 + 1.0);
      let H = normalize(V + L);
      let NdotL = max(dot(N, L), 0.0);
      let shadow = calcSoftshadow(pos + N * 0.02, L, 8.0);
      var occ = 1.0;
      if (objMat != 2) { occ = calcAO(pos, N); } 

      var D = 0.0;
      if (!isRightSide && objMat == 2) { 
          D = DistributionGTR(N, H, roughness, u.gtrGamma);
      } else {
          D = DistributionGGX(N, H, roughness);
      }

      let G = GeometrySmith(N, V, L, roughness);
      let F = fresnelSchlick(max(dot(H, V), 0.0), F0);
      
      let kS = F;
      var kD = (vec3f(1.0) - kS) * (1.0 - metallic); 
      var specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
      
      if (!isRightSide && objMat == 2) {
         let energyComp = EnergyCompensation(F0, roughness);
         specular *= energyComp; 
         kD *= (1.0 / energyComp);
      }

      if (u.iridescence > 0.0 && objMat == 2 && !isRightSide) {
           baseColor = bumpIridescence(baseColor, NdotV, 3.0);
      }
      
      let lightColor = vec3f(1.0, 0.95, 0.8) * 4.0; 
      var directLight = (kD * baseColor / PI + specular) * NdotL * shadow * attenuation * lightColor;
      
      if (u.sssStrength > 0.0 && metallic < 0.1 && objMat == 2) {
          let sss = getSubsurfaceScattering(pos, N, L, V);
          let sssTint = vec3f(1.0, 0.3, 0.1);
          directLight += sss * sssTint * u.sssStrength * attenuation * 3.0;
      }
      
      color += directLight;

      if (transmission > 0.0) {
          let safeIor = max(u.ior, 1.0);
          let refrVec = refract(rd, N, 1.0 / safeIor);
          let thickness = 2.0 * (1.0 + NdotV); 
          let absorb = exp(-vec3f(0.15, 0.05, 0.02) * thickness); 
          let env = getEnvironment(refrVec, 0.0) * absorb;
          let fresnel = fresnelSchlick(NdotV, F0);
          color = mix(env * baseColor, color, fresnel.x);
      } 
      
      var reflectionColor = getEnvironment(R, roughness); 
      if (roughness < 0.6) {
          var tRef = 0.05;
          var hitRef = false;
          for(var j=0; j<48; j++) {
              let h = map(pos + R * tRef);
              if (h < EPSILON) { hitRef = true; break; }
              tRef += h;
              if (tRef > MAX_DIST) { break; }
          }
          if (hitRef) { reflectionColor = vec3f(0.05) * lightColor; }
          reflectionColor = mix(reflectionColor, getEnvironment(R, 1.0), roughness);
      }
      color += reflectionColor * F * occ;
      
      if (u.clearcoat > 0.0 && objMat == 2 && !isRightSide) {
           let F_cc = fresnelSchlick(NdotV, vec3f(0.04)) * u.clearcoat;
           color += getEnvironment(R, 0.0) * F_cc;
      }

      // WIREFRAME OVERLAY
      if (objMat == 2 && u.wireframe > 0.0) {
          let minDist = min(min(bary.x, bary.y), bary.z);
          let edgeWidth = 0.02;
          let edge = smoothstep(edgeWidth, 0.0, minDist);
          let wireColor = vec3f(0.0, 1.0, 0.5) * 5.0; // Neon Green
          color = mix(color, wireColor, edge * u.wireframe);
      }
      
  } else {
      color = getEnvironment(rd, 0.0);
  }
  
  if (u.volumetricDensity > 0.0) {
      let lightDir = normalize(lightPos - ro); 
      let fog = getVolumetricLight(ro, rd, lightDir, min(t, MAX_DIST));
      color += vec3f(0.8, 0.85, 0.9) * fog;
  }
  
  let distToLine = abs(p_screen.x - splitX);
  if (distToLine < 0.005) { color = vec3f(0.8, 1.0, 0.0); } 
  let fringeStr = 0.006;
  let fringe = smoothstep(0.4, 0.9, dist) * fringeStr;
  color.r += fringe;
  color.b -= fringe;
  color *= 0.8; 
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  color = clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
  let vig = 1.0 - length(uv_distorted - 0.5) * 0.5;
  color *= vig;
  color = pow(color, vec3f(1.0 / 2.2));
  let ign = interleavedGradientNoise(uv * u.resolution);
  color += (ign - 0.5) / 255.0;

  return vec4f(color, 1.0);
}
`;