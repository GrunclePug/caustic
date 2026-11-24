// Pass 2: Resolve fixed-point atomic<u32> spectral CIE XYZ buffer into linear Rec.709 HDR,
// clear atomic deposits for the next frame, apply accumulation or pulse persistence decay,
// and draw subtle optical element, GRIN medium, or Kerr horizon-ergosphere guides.

struct ResolveUniforms {
    // xy = view_center, z = view_half_height, w = aspect
    view: vec4<f32>,
    // x = inv_fixed_scale, y = decay (Pulse mode), z = exposure_gain, w = accum_norm (1 / accum_frames)
    params: vec4<f32>,
    // x = spin a, y = r_plus, z = grin_strength, w = dispersion_scale
    kerr: vec4<f32>,
    // x = width, y = height, z = physics_kind, w = mode (0=Accumulate, 1=Pulse)
    dims: vec4<u32>,
    // x = num_segments, y = num_circles, z = clear_hdr (0/1), w = reserved
    counts: vec4<u32>,
};

struct OptSegment {
    endpoints: vec4<f32>,
    material: vec4<f32>,
    info: vec4<u32>,
};

struct OptCircle {
    geom: vec4<f32>,
    clip_arc: vec4<f32>,
    material: vec4<f32>,
    info: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: ResolveUniforms;
@group(0) @binding(1) var<storage, read_write> accum_xyz: array<u32>;
@group(0) @binding(2) var<storage, read_write> hdr_accum: array<vec4<f32>>;
@group(0) @binding(3) var hdr_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<storage, read> segments: array<OptSegment>;
@group(0) @binding(5) var<storage, read> circles: array<OptCircle>;

const PHYS_OPTICS: u32 = 0u;
const PHYS_KERR: u32 = 1u;
const PHYS_GRIN_LUNEBURG: u32 = 2u;
const PHYS_GRIN_BH: u32 = 3u;
const PHYS_GRIN_MIRAGE: u32 = 4u;

const MAT_MIRROR: u32 = 1u;
const MAT_ABSORBER: u32 = 3u;

const LUNEBURG_R: f32 = 3.2;
const OPT_BH_BC_BASE: f32 = 2.4;
const OPT_BH_CORE_R: f32 = 0.72;
const MIRAGE_Y_GROUND: f32 = -2.2;
const MIRAGE_SCALE_H: f32 = 1.35;

fn xyz_to_rec709(xyz: vec3<f32>) -> vec3<f32> {
    let rgb = vec3<f32>(
         3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z,
        -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z,
         0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z,
    );
    let min_c = min(rgb.r, min(rgb.g, rgb.b));
    if (min_c < 0.0) {
        let max_c = max(rgb.r, max(rgb.g, rgb.b));
        let pivot = max(max(xyz.y, 0.0), max_c * 0.14);
        let alpha = pivot / max(pivot - min_c, 1e-6);
        return max(mix(vec3<f32>(pivot), rgb, clamp(alpha, 0.0, 1.0)), vec3<f32>(0.0));
    }
    return rgb;
}

fn dist_to_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-10), 0.0, 1.0);
    return length(p - (a + ab * t));
}

fn scene_overlay(world: vec2<f32>, px_world: f32) -> vec3<f32> {
    var col = vec3<f32>(0.0);
    let line_w = px_world * 1.4;
    let phys = u.dims.z;

    if (phys == PHYS_KERR) {
        let a = u.kerr.x;
        let rp = u.kerr.y;
        let r = sqrt(max(dot(world, world) - a * a, 0.0));
        let d_hor = abs(r - rp);
        col = col + vec3<f32>(0.18, 0.22, 0.32) * exp(-(d_hor * d_hor) / (line_w * line_w * 2.0));
        let d_ergo = abs(r - 2.0);
        col = col + vec3<f32>(0.04, 0.09, 0.16) * exp(-(d_ergo * d_ergo) / (line_w * line_w * 1.5));
        return col;
    }

    if (phys == PHYS_GRIN_LUNEBURG) {
        let r = length(world);
        let d_rim = abs(r - LUNEBURG_R);
        col = col + vec3<f32>(0.10, 0.20, 0.30) * exp(-(d_rim * d_rim) / (line_w * line_w * 1.4));
        if (r < LUNEBURG_R) {
            let grad_vis = (1.0 - (r * r) / (LUNEBURG_R * LUNEBURG_R));
            col = col + vec3<f32>(0.008, 0.018, 0.032) * grad_vis;
        }
        return col;
    }

    if (phys == PHYS_GRIN_BH) {
        let r = length(world);
        let d_core = abs(r - OPT_BH_CORE_R);
        col = col + vec3<f32>(0.22, 0.26, 0.36) * exp(-(d_core * d_core) / (line_w * line_w * 1.8));
        let bc = OPT_BH_BC_BASE * sqrt(max(u.kerr.z, 0.05));
        let d_ph = abs(r - bc);
        col = col + vec3<f32>(0.05, 0.09, 0.14) * exp(-(d_ph * d_ph) / (line_w * line_w * 1.4));
        return col;
    }

    if (phys == PHYS_GRIN_MIRAGE) {
        let d_gnd = abs(world.y - MIRAGE_Y_GROUND);
        col = col + vec3<f32>(0.28, 0.14, 0.06) * exp(-(d_gnd * d_gnd) / (line_w * line_w * 1.8));
        if (world.y > MIRAGE_Y_GROUND) {
            let heat = exp(-(world.y - MIRAGE_Y_GROUND) / MIRAGE_SCALE_H);
            col = col + vec3<f32>(0.025, 0.008, 0.002) * heat;
        }
        return col;
    }

    // PHYS_OPTICS wireframe guides for glass, mirrors, and beam stops.
    for (var i = 0u; i < u.counts.x; i = i + 1u) {
        let seg = segments[i];
        let d = dist_to_segment(world, seg.endpoints.xy, seg.endpoints.zw);
        let alpha = exp(-(d * d) / (line_w * line_w));
        var tint = vec3<f32>(0.08, 0.16, 0.24);
        if (seg.info.x == MAT_MIRROR) {
            tint = vec3<f32>(0.25, 0.28, 0.34);
        } else if (seg.info.x == MAT_ABSORBER) {
            tint = vec3<f32>(0.34, 0.12, 0.10);
        }
        col = col + tint * alpha;
    }

    for (var i = 0u; i < u.counts.y; i = i + 1u) {
        let cir = circles[i];
        let center = cir.geom.xy;
        let radius = cir.geom.z;
        let clip_r = cir.geom.w;
        let clip_c = cir.clip_arc.xy;
        let arc_cos = cir.clip_arc.z;
        let arc_ang = cir.clip_arc.w;
        let arc_dir = vec2<f32>(cos(arc_ang), sin(arc_ang));

        let v = world - center;
        let len = length(v);
        if (len < 1e-6) {
            continue;
        }
        let n_out = v / len;
        if (clip_r > 0.0) {
            let dc = world - clip_c;
            if (dot(dc, dc) > (clip_r + line_w) * (clip_r + line_w)) {
                continue;
            }
        }
        if (arc_cos > -1.5 && dot(n_out, arc_dir) < arc_cos) {
            continue;
        }
        let d = abs(len - radius);
        let alpha = exp(-(d * d) / (line_w * line_w));
        let tint = select(vec3<f32>(0.08, 0.16, 0.24), vec3<f32>(0.28, 0.32, 0.38), cir.info.x == MAT_MIRROR);
        col = col + tint * alpha;
    }

    return col;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = u.dims.x;
    let height = u.dims.y;
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let idx = gid.y * width + gid.x;
    let base = idx * 3u;

    let rx = f32(accum_xyz[base]) * u.params.x;
    let ry = f32(accum_xyz[base + 1u]) * u.params.x;
    let rz = f32(accum_xyz[base + 2u]) * u.params.x;
    accum_xyz[base] = 0u;
    accum_xyz[base + 1u] = 0u;
    accum_xyz[base + 2u] = 0u;

    let frame_xyz = vec3<f32>(rx, ry, rz);
    var acc_xyz = hdr_accum[idx].xyz;
    if (u.counts.z != 0u) {
        acc_xyz = frame_xyz;
    } else if (u.dims.w == 0u) {
        acc_xyz = acc_xyz + frame_xyz;
    } else {
        acc_xyz = acc_xyz * u.params.y + frame_xyz;
    }
    hdr_accum[idx] = vec4<f32>(acc_xyz, 1.0);

    let scale = select(u.params.z, u.params.z * u.params.w, u.dims.w == 0u);
    var rgb = xyz_to_rec709(acc_xyz * scale);

    let half_h = u.view.z;
    let half_w = half_h * u.view.w;
    let uv = (vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5)) / vec2<f32>(f32(width), f32(height));
    let world = u.view.xy + vec2<f32>((uv.x * 2.0 - 1.0) * half_w, (1.0 - uv.y * 2.0) * half_h);
    let px_world = (2.0 * half_h) / f32(height);

    let phys = u.dims.z;
    if (phys == PHYS_KERR) {
        let a = u.kerr.x;
        let rp = u.kerr.y;
        let r = sqrt(max(dot(world, world) - a * a, 0.0));
        if (r < rp) {
            rgb = vec3<f32>(0.0);
        }
    } else if (phys == PHYS_GRIN_BH) {
        if (dot(world, world) < OPT_BH_CORE_R * OPT_BH_CORE_R) {
            rgb = vec3<f32>(0.0);
        }
    } else if (phys == PHYS_GRIN_MIRAGE) {
        if (world.y < MIRAGE_Y_GROUND) {
            rgb = vec3<f32>(0.0);
        }
    }
    rgb = rgb + scene_overlay(world, px_world);

    textureStore(hdr_out, vec2<i32>(gid.xy), vec4<f32>(min(rgb, vec3<f32>(60000.0)), 1.0));
}
