// Pass 2: Resolve fixed-point atomic<u32> spectral CIE XYZ buffer into linear Rec.709 HDR,
// clear atomic deposits for the next frame, apply accumulation or pulse persistence decay,
// and render optical elements, real 2K textured 3D-lit planets/moons/Sun with eclipse shadows,
// Cassini-banded rings, GRIN fields, and Kerr horizons.

struct ResolveUniforms {
    // xy = view_center, z = view_half_height, w = aspect
    view: vec4<f32>,
    // x = inv_fixed_scale, y = decay (Pulse mode), z = exposure_gain, w = accum_norm (1 / accum_frames)
    params: vec4<f32>,
    // x = spin a, y = r_plus, z = grin_strength, w = dispersion_scale
    kerr: vec4<f32>,
    // Optional rogue Schwarzschild black hole: xy = pos, z = geometric mass M (<=0 disables), w = reserved
    bh: vec4<f32>,
    // x = width, y = height, z = physics_kind, w = mode (0=Accumulate, 1=Pulse)
    dims: vec4<u32>,
    // x = num_segments, y = num_circles, z = clear_hdr (0/1/2), w = num_emitters
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

struct GpuEmitter {
    geom: vec4<f32>,
    spec: vec4<f32>,
    info: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: ResolveUniforms;
@group(0) @binding(1) var<storage, read_write> accum_xyz: array<u32>;
@group(0) @binding(2) var<storage, read_write> hdr_accum: array<vec4<f32>>;
@group(0) @binding(3) var hdr_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<storage, read> segments: array<OptSegment>;
@group(0) @binding(5) var<storage, read> circles: array<OptCircle>;
@group(0) @binding(6) var<storage, read> emitters: array<GpuEmitter>;
@group(0) @binding(7) var planet_tex: texture_2d_array<f32>;
@group(0) @binding(8) var planet_sampler: sampler;
@group(0) @binding(9) var ring_tex: texture_2d<f32>;

const PI: f32 = 3.14159265;
const TAU: f32 = 6.28318530;

const PHYS_OPTICS: u32 = 0u;
const PHYS_KERR: u32 = 1u;
const PHYS_GRIN_LUNEBURG: u32 = 2u;
const PHYS_GRIN_BH: u32 = 3u;
const PHYS_GRIN_MIRAGE: u32 = 4u;

const MAT_MIRROR: u32 = 1u;
const MAT_ABSORBER: u32 = 3u;
const MAT_DIFFUSE: u32 = 4u;
const MAT_ATMOSPHERE: u32 = 5u;
const MAT_SPLITTER: u32 = 6u;

const EMITTER_DISK: u32 = 3u;
const EMITTER_SOLAR_FAR: u32 = 4u;
const SPEC_BLACKBODY: u32 = 1u;

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

fn star_color_tint(temp_k: f32) -> vec3<f32> {
    let t = clamp((temp_k - 2800.0) / 9700.0, 0.0, 1.0);
    let red = vec3<f32>(1.0, 0.34, 0.10);
    let sun = vec3<f32>(1.0, 0.94, 0.82);
    let blue = vec3<f32>(0.42, 0.72, 1.0);
    return select(mix(sun, blue, (t - 0.31) / 0.69), mix(red, sun, t / 0.31), t < 0.31);
}

// Map 2K solar granulation texture cleanly across spectral star classes (Red Dwarf -> Sun -> Blue Giant).
fn star_photosphere_color(sun_tex: vec3<f32>, temp_k: f32, lz: f32) -> vec3<f32> {
    let limb = 0.42 + 0.58 * pow(lz, 0.55);
    let gran = clamp(dot(sun_tex, vec3<f32>(0.30, 0.58, 0.12)), 0.0, 1.0);
    if (temp_k < 4600.0) {
        let f = clamp((4600.0 - temp_k) / 1600.0, 0.0, 1.0);
        let red_plasma = mix(vec3<f32>(0.42, 0.03, 0.005), vec3<f32>(1.0, 0.42, 0.09), pow(gran, 0.72));
        let base = mix(sun_tex, red_plasma, f);
        return base * (1.10 + 1.35 * limb) + vec3<f32>(0.22, 0.05, 0.01) * (lz * lz);
    }
    if (temp_k > 7500.0) {
        let f = clamp((temp_k - 7500.0) / 4500.0, 0.0, 1.0);
        let blue_plasma = mix(vec3<f32>(0.08, 0.24, 0.78), vec3<f32>(0.80, 0.94, 1.0), pow(gran, 0.68));
        let base = mix(sun_tex, blue_plasma, f);
        return base * (1.20 + 1.50 * limb) + vec3<f32>(0.10, 0.18, 0.32) * (lz * lz);
    }
    return sun_tex * (1.25 + 1.45 * limb) + vec3<f32>(0.22, 0.11, 0.025) * (lz * lz);
}

// Returns xy = unit vector from `pt` toward emitter `em_idx`, z = angular radius (rad), w = distance.
fn emitter_light_info(em_idx: u32, pt: vec2<f32>) -> vec4<f32> {
    let em = emitters[em_idx];
    if (em.info.x == EMITTER_SOLAR_FAR) {
        let sdir = vec2<f32>(cos(em.geom.z), sin(em.geom.z));
        return vec4<f32>(-sdir, max(em.spec.w, 1e-4), 1e6);
    }
    let to_sun = em.geom.xy - pt;
    let dist = max(length(to_sun), 1e-5);
    let dir = to_sun / dist;
    let rad = select(0.03, max(em.geom.w, 1e-5), em.info.x == EMITTER_DISK);
    let ang = atan2(rad, max(dist, rad));
    return vec4<f32>(dir, ang, dist);
}

// Compute real-time 3D RGB solar illumination at 3D sphere point `(pt, z_height)` for ecliptic sunlight,
// accounting for 3D spherical umbra/penumbra eclipses, Rayleigh red lunar eclipses, and Cassini ring shadows.
fn eclipse_illumination(pt: vec2<f32>, z_height: f32, self_idx: u32, sun_dir: vec2<f32>, sun_ang_r: f32, sun_dist: f32) -> vec3<f32> {
    var illum = vec3<f32>(1.0);
    let perp_axis = vec2<f32>(-sun_dir.y, sun_dir.x);

    for (var j = 0u; j < u.counts.y; j = j + 1u) {
        if (j == self_idx) {
            continue;
        }
        let occ = circles[j];
        if (occ.info.x != MAT_DIFFUSE) {
            continue;
        }
        let d = occ.geom.xy - pt;
        let t_along = dot(d, sun_dir);
        if (t_along <= 0.0 || t_along >= sun_dist) {
            continue;
        }
        let perp_xy = dot(d, perp_axis);
        let perp_3d = sqrt(perp_xy * perp_xy + z_height * z_height);
        let occ_ang = occ.geom.z / t_along;
        let sep_ang = perp_3d / t_along;
        if (sep_ang < occ_ang + sun_ang_r) {
            let edge_in = max(occ_ang - sun_ang_r, 0.0);
            let edge_out = occ_ang + sun_ang_r;
            let frac = 1.0 - smoothstep(edge_in, edge_out, sep_ang);
            let max_cov = min((occ_ang * occ_ang) / max(sun_ang_r * sun_ang_r, 1e-8), 1.0);
            let cov = frac * max_cov;
            let has_atmo = (j + 1u < u.counts.y) && (circles[j + 1u].info.x == MAT_ATMOSPHERE);
            if (has_atmo) {
                // Refracted sunset copper-red glow inside an atmospheric planet's lunar umbra.
                let umbra_tint = vec3<f32>(0.44, 0.07, 0.02);
                illum = illum * mix(vec3<f32>(1.0), umbra_tint, cov);
            } else {
                // Crisp dark umbra cast by airless moons (Moon on Earth, Io on Jupiter).
                illum = illum * (1.0 - cov * 0.96);
            }
        }
    }

    // Translucent Cassini-banded shadow cast by upstream planetary ring segments.
    for (var s = 0u; s < u.counts.x; s = s + 1u) {
        let seg = segments[s];
        if (seg.info.x != MAT_SPLITTER || seg.info.y != 1u) {
            continue;
        }
        let a = seg.endpoints.xy;
        let ab = seg.endpoints.zw - a;
        let denom = sun_dir.x * ab.y - sun_dir.y * ab.x;
        if (abs(denom) < 1e-7) {
            continue;
        }
        let ap = a - pt;
        let t_ray = (ap.x * ab.y - ap.y * ab.x) / denom;
        let u_seg = (ap.x * sun_dir.y - ap.y * sun_dir.x) / denom;
        if (t_ray > 1e-4 && t_ray < sun_dist && u_seg >= 0.0 && u_seg <= 1.0) {
            let ring_s = textureSampleLevel(ring_tex, planet_sampler, vec2<f32>(clamp(u_seg, 0.002, 0.998), 0.5), 0.0);
            illum = illum * (1.0 - ring_s.a * 0.62);
        }
    }

    return illum;
}

// Sample a 2K equirectangular planet/Sun texture on a 3D unit hemisphere with analytical mipmap LOD.
fn sample_sphere_tex(lp: vec2<f32>, lz: f32, spin_angle: f32, tilt: f32, layer: i32, rad_px: f32) -> vec3<f32> {
    let ct = cos(tilt);
    let st = sin(tilt);
    let nx = lp.x;
    let ny = lp.y * ct + lz * st;
    let nz = -lp.y * st + lz * ct;
    let lon = atan2(nx, nz) + spin_angle;
    let lat = asin(clamp(ny, -1.0, 1.0));
    let u_tex = fract(lon / TAU + 0.5);
    let v_tex = clamp(0.5 - lat / PI, 0.001, 0.999);
    let tex_w = f32(textureDimensions(planet_tex).x);
    let lod = clamp(log2(tex_w / max(TAU * rad_px, 1.0)), 0.0, 10.0);
    return textureSampleLevel(planet_tex, planet_sampler, vec2<f32>(u_tex, v_tex), layer, lod).rgb;
}

fn scene_overlay(world: vec2<f32>, px_world: f32, base_rgb: vec3<f32>) -> vec3<f32> {
    var col = base_rgb;
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

    // Optical segments: glass, mirrors, beam splitters, Cassini-banded icy rings, matte walls, absorbers.
    for (var i = 0u; i < u.counts.x; i = i + 1u) {
        let seg = segments[i];
        let a = seg.endpoints.xy;
        let b = seg.endpoints.zw;
        let ab = b - a;
        let ab_len2 = max(dot(ab, ab), 1e-10);
        let t_seg = clamp(dot(world - a, ab) / ab_len2, 0.0, 1.0);
        let d = length(world - (a + ab * t_seg));

        if (seg.info.x == MAT_SPLITTER && seg.info.y == 1u) {
            // Real Cassini radial ring banding (C, B, Cassini Division, A rings) illuminated by all stars.
            let ring_w = max(px_world * 2.2, sqrt(ab_len2) * 0.052);
            if (d < ring_w * 3.0) {
                let ring_s = textureSampleLevel(
                    ring_tex,
                    planet_sampler,
                    vec2<f32>(clamp(t_seg, 0.002, 0.998), 0.5),
                    0.0,
                );
                let n_em = min(u.counts.w, 4u);
                var light_sum = vec3<f32>(0.0);
                let em_scale = 1.0 / max(f32(n_em), 1.0);
                for (var e = 0u; e < n_em; e = e + 1u) {
                    let em = emitters[e];
                    let sinfo = emitter_light_info(e, world);
                    let ecl = eclipse_illumination(world, 0.0, 9999u, sinfo.xy, sinfo.z, sinfo.w);
                    let st = select(vec3<f32>(1.0), star_color_tint(em.spec.x), em.info.y == SPEC_BLACKBODY);
                    light_sum = light_sum + st * ecl * em_scale;
                }
                let prof = exp(-(d * d) / (ring_w * ring_w));
                let ring_lit = ring_s.rgb * (1.35 * light_sum + vec3<f32>(0.05));
                col = col + ring_lit * (ring_s.a * prof);
            }
            continue;
        }

        let alpha = exp(-(d * d) / (line_w * line_w));
        var tint = vec3<f32>(0.08, 0.16, 0.24);
        if (seg.info.x == MAT_MIRROR) {
            tint = vec3<f32>(0.26, 0.30, 0.36);
        } else if (seg.info.x == MAT_ABSORBER) {
            tint = vec3<f32>(0.34, 0.12, 0.10);
        } else if (seg.info.x == MAT_DIFFUSE) {
            tint = seg.material.xyz * 0.28 + vec3<f32>(0.05);
        } else if (seg.info.x == MAT_SPLITTER) {
            tint = vec3<f32>(0.16, 0.32, 0.40);
        }
        col = col + tint * alpha;
    }

    // Glowing star / Sun disks for EMITTER_DISK emitters (with 2K solar photosphere granulation when tagged).
    for (var i = 0u; i < u.counts.w; i = i + 1u) {
        let em = emitters[i];
        if (em.info.x == EMITTER_DISK) {
            let v = world - em.geom.xy;
            let r = length(v);
            let true_rad = max(em.geom.w, 1e-6);
            let vis_rad = max(true_rad, px_world * 2.5);
            let tint = select(vec3<f32>(1.0, 0.96, 0.88), star_color_tint(em.spec.x), em.info.y == SPEC_BLACKBODY);
            let rad_px = vis_rad / px_world;
            if (r <= vis_rad + px_world) {
                let cov = 1.0 - smoothstep(vis_rad - px_world * 0.75, vis_rad + px_world * 0.75, r);
                let lp = v / vis_rad;
                let lz = sqrt(max(1.0 - min(dot(lp, lp), 1.0), 0.0));
                if (em.info.z == 1u && rad_px > 5.0) {
                    let sun_tex = sample_sphere_tex(lp, lz, em.spec.w, 0.18, 0, rad_px);
                    let photosphere = star_photosphere_color(sun_tex, em.spec.x, lz);
                    col = mix(col, photosphere, cov);
                } else {
                    let limb = 0.45 + 0.55 * lz;
                    col = mix(col, tint * (3.6 * limb), cov);
                }
            }
            if (r > vis_rad - px_world) {
                let d = max(r - vis_rad, 0.0);
                let halo_w = max(true_rad * 0.30, px_world * 10.0);
                let corona_tint = select(tint, mix(tint, vec3<f32>(1.0, 0.80, 0.48), 0.25), em.info.z == 1u && rad_px > 5.0);
                col = col + corona_tint * (0.42 * exp(-d / halo_w));
            }
        }
    }

    // Circular optics, 2K textured 3D-lit planets/moons, and subtle atmospheric shells.
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
        if (clip_r > 0.0) {
            let dc = world - clip_c;
            if (dot(dc, dc) > (clip_r + line_w) * (clip_r + line_w)) {
                continue;
            }
        }
        if (arc_cos > -1.5 && len > 1e-6 && dot(v / len, arc_dir) < arc_cos) {
            continue;
        }

        if (cir.info.x == MAT_DIFFUSE) {
            if (cir.info.y > 0u) {
                if (len <= radius + px_world) {
                    let cov = 1.0 - smoothstep(radius - px_world * 0.75, radius + px_world * 0.75, len);
                    let lp = v / max(radius, 1e-6);
                    let lz = sqrt(max(1.0 - min(dot(lp, lp), 1.0), 0.0));
                    let rad_px = radius / px_world;
                    let layer = i32(cir.info.y - 1u);
                    let tex_rgb = sample_sphere_tex(lp, lz, cir.material.w, 0.28, layer, rad_px);

                    let n_em = min(u.counts.w, 4u);
                    let has_atmo = (i + 1u < u.counts.y) && (circles[i + 1u].info.x == MAT_ATMOSPHERE);
                    let rim_base = mix(cir.material.xyz, vec3<f32>(0.26, 0.58, 1.0), 0.45);
                    let em_norm = select(1.0, 0.85, n_em >= 2u);
                    var direct_light = vec3<f32>(0.0);
                    var atmo_light = vec3<f32>(0.0);
                    if (n_em == 0u) {
                        let sun_3d = normalize(vec3<f32>(-1.0, 0.0, 0.22));
                        let ndotl = max(dot(vec3<f32>(lp.x, lp.y, lz), sun_3d), 0.0);
                        direct_light = vec3<f32>(pow(ndotl, 0.85));
                    } else {
                        for (var e = 0u; e < n_em; e = e + 1u) {
                            let em = emitters[e];
                            let sinfo = emitter_light_info(e, center);
                            let sun_3d = normalize(vec3<f32>(sinfo.x, sinfo.y, 0.22));
                            let ndotl = max(dot(vec3<f32>(lp.x, lp.y, lz), sun_3d), 0.0);
                            let lit = pow(ndotl, 0.85);
                            let ecl = eclipse_illumination(world, radius * lz, i, sinfo.xy, sinfo.z, sinfo.w);
                            let st = select(vec3<f32>(1.0), star_color_tint(em.spec.x), em.info.y == SPEC_BLACKBODY);
                            direct_light = direct_light + st * (lit * em_norm) * ecl;
                            if (has_atmo) {
                                let fres = pow(1.0 - lz, 2.8) * smoothstep(0.0, 0.40, ndotl + 0.14);
                                atmo_light = atmo_light + (rim_base * st) * (0.42 * fres * em_norm) * ecl;
                            }
                        }
                    }

                    let planet_col = tex_rgb * (1.28 * direct_light + vec3<f32>(0.032)) + atmo_light;
                    col = mix(col, planet_col, cov);
                }
            } else {
                let d = abs(len - radius);
                let alpha = exp(-(d * d) / (line_w * line_w));
                if (len < radius) {
                    col = col + cir.material.xyz * 0.055;
                }
                col = col + (cir.material.xyz * 0.28 + vec3<f32>(0.04)) * alpha;
            }
        } else if (cir.info.x == MAT_ATMOSPHERE) {
            let d = abs(len - radius);
            let alpha = exp(-(d * d) / (line_w * line_w));
            col = col + vec3<f32>(0.005, 0.014, 0.034) * alpha;
        } else {
            let d = abs(len - radius);
            let alpha = exp(-(d * d) / (line_w * line_w));
            let tint = select(vec3<f32>(0.08, 0.16, 0.24), vec3<f32>(0.28, 0.32, 0.38), cir.info.x == MAT_MIRROR);
            col = col + tint * alpha;
        }
    }

    // Optional drop-in rogue Schwarzschild black hole: abyssal horizon core + subtle Einstein photon ring.
    if (u.bh.z > 0.0) {
        let bh_m = u.bh.z;
        let r_bh = length(world - u.bh.xy);
        let r_hor = max(0.64 * bh_m, px_world * 2.5);
        let r_ph = 1.8660254 * bh_m;
        let hor_cov = 1.0 - smoothstep(r_hor - px_world, r_hor + px_world, r_bh);
        col = mix(col, vec3<f32>(0.0), hor_cov);
        let d_ph = abs(r_bh - r_ph);
        let ph_w = max(px_world * 1.4, bh_m * 0.07);
        col = col + vec3<f32>(0.45, 0.65, 1.0) * (0.20 * exp(-(d_ph * d_ph) / (ph_w * ph_w)));
        let d_rim = abs(r_bh - r_hor);
        col = col + vec3<f32>(0.95, 0.78, 0.48) * (0.32 * exp(-(d_rim * d_rim) / (line_w * line_w * 1.8)));
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
    if (u.counts.z == 1u) {
        acc_xyz = frame_xyz;
    } else if (u.counts.z == 2u) {
        // Temporal EMA for smoothly orbiting / animated Steady scenes (zero speckle, zero lag).
        let prev_avg = acc_xyz * u.params.w;
        acc_xyz = mix(prev_avg, frame_xyz, 0.06);
    } else if (u.dims.w == 0u) {
        acc_xyz = acc_xyz + frame_xyz;
    } else {
        acc_xyz = acc_xyz * u.params.y + frame_xyz;
    }
    hdr_accum[idx] = vec4<f32>(acc_xyz, 1.0);

    let eff_norm = select(u.params.w, 1.0, u.counts.z != 0u);
    let scale = select(u.params.z, u.params.z * eff_norm, u.dims.w == 0u);
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
    } else {
        // Mask solid untextured planet interiors (textured planets composite smoothly over photon splats).
        for (var i = 0u; i < u.counts.y; i = i + 1u) {
            let cir = circles[i];
            if (cir.info.x == MAT_DIFFUSE && cir.info.y == 0u) {
                let v = world - cir.geom.xy;
                if (dot(v, v) < (cir.geom.z - px_world * 0.5) * (cir.geom.z - px_world * 0.5)) {
                    rgb = vec3<f32>(0.0);
                }
            }
        }
    }
    rgb = scene_overlay(world, px_world, rgb);

    textureStore(hdr_out, vec2<i32>(gid.xy), vec4<f32>(min(rgb, vec3<f32>(60000.0)), 1.0));
}
