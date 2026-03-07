use std::rc::Rc;

use xenofrost::core::app::App;
use xenofrost::core::input_manager::{InputManager, KeyCode};
use xenofrost::core::math::bounding2d::Polygon2d;
use xenofrost::core::render_engine::buffer::{Buffer, VecBuffer};
use xenofrost::core::render_engine::gui::font_renderer::{CharacterInstance, DefaultFonts, FontSpecification, SdfCharacterInstance, construct_sdf_string_instance_data, construct_string_instance_data, create_bitmap_font_pipeline, create_font_ratio_bind_group_layout, create_sdf_aemrange_bind_group_layout, create_sdf_font_pipeline, get_font_from_defaults, get_font_ratio, get_sdf_aem_distance_bind_group};
use xenofrost::core::render_engine::mesh::{Mesh, create_atlas_quad_mesh};
use xenofrost::core::render_engine::pipeline::{InstanceAtlas, InstanceDebugLine, create_aspect_ratio_bind_group_layout, create_atlas_pipeline2d, create_color_bind_group_layout, create_debug_lines_pipeline2d};
use xenofrost::core::render_engine::render_camera::{RenderCamera, create_camera_bind_group_layout};
use xenofrost::core::render_engine::texture::{Texture, TextureAtlasUtil, TextureCoordUtil, create_texture_bind_group, create_texture_bind_group_layout};
use xenofrost::core::render_engine::wgpu::BufferUsages;
use xenofrost::core::render_engine::{DrawMesh, bytemuck, wgpu};
use xenofrost::core::render_engine::{RenderEngine, create_command_encoder};
use xenofrost::core::math::{IVec2, Mat4, Quat, Transform2d, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles};
use xenofrost::core::utilities::WorldVec;
use xenofrost::core::world::{Animation2d, AnimationFrame2d, AnimationObject2d};
use xenofrost::core::world::camera::{Camera2d, CameraProjection, OrthographicProjection};
use xenofrost::include_bytes_from_project_path;

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

const TANK_COLLISION_POINTS: &[&[Vec2]] = &[
    &[Vec2::new(-0.2890625, 0.21875), Vec2::new(-0.375, 0.1171875), Vec2::new(-0.4453125, -0.21875), Vec2::new(-0.328125, -0.328125), Vec2::new(0.3125, -0.328125), Vec2::new(0.4375, -0.234375), Vec2::new(0.375, 0.09375), Vec2::new(0.2578125, 0.21875)],
    &[Vec2::new(-0.46875, 0.0), Vec2::new(-0.3125, -0.3203125), Vec2::new(-0.1953125, -0.390625), Vec2::new(0.3828125, -0.2578125), Vec2::new(0.484375, -0.140625), Vec2::new(0.484375, -0.0546875), Vec2::new(0.15625, 0.25), Vec2::new(-0.3671875, 0.1796875)],
    &[Vec2::new(-0.4921875, -0.1328125), Vec2::new(-0.078125, -0.40625), Vec2::new(-0.015625, -0.40625), Vec2::new(0.390625, -0.1640625), Vec2::new(0.453125, -0.046875), Vec2::new(0.4609375, 0.0625), Vec2::new(0.390625, 0.1171875), Vec2::new(0.0078125, 0.2578125), Vec2::new(-0.40625, 0.109375), Vec2::new(-0.5, -0.03125)],
    &[Vec2::new(-0.3984375, -0.2890625), Vec2::new(0.1796875, -0.390625), Vec2::new(0.296875, -0.0703125), Vec2::new(0.34375, 0.1875), Vec2::new(-0.1875, 0.2421875), Vec2::new(-0.3828125, 0.0078125), Vec2::new(-0.421875, -0.1875)],
    &[Vec2::new(-0.3359375, -0.2265625), Vec2::new(-0.3046875, -0.34375), Vec2::new(0.296875, -0.34375), Vec2::new(0.3125, -0.2265625), Vec2::new(0.2578125, 0.21875), Vec2::new(-0.28125, 0.21875)],
    &[Vec2::new(-0.3359375, 0.1953125), Vec2::new(-0.2109375, -0.375), Vec2::new(0.375, -0.3046875), Vec2::new(0.359375, 0.0), Vec2::new(0.1875, 0.234375)],
    &[Vec2::new(-0.4765625, 0.0234375), Vec2::new(-0.421875, -0.125), Vec2::new(0.0078125, -0.40625), Vec2::new(0.1015625, -0.3828125),Vec2::new(0.484375, -0.125),Vec2::new(0.484375, -0.046875), Vec2::new(0.40625, 0.09375), Vec2::new(0.0, 0.2578125), Vec2::new(-0.4140625, 0.109375)],
    &[Vec2::new(-0.390625, -0.265625), Vec2::new(0.1953125, -0.3828125), Vec2::new(0.3046875, -0.3046875), Vec2::new(0.4609375, 0.0078125), Vec2::new(0.3515625, 0.171875), Vec2::new(-0.1640625, 0.2421875), Vec2::new(-0.5, -0.0703125),Vec2::new(-0.4921875, -0.15625)],
];

const BULLET_WIDTH: f32 = 0.25;
const BULLET_HEIGHT: f32 = 0.088;
const BULLET_COLLISION_POINTS: &[Vec2] = &[Vec2::new(-1.0*BULLET_WIDTH/2.0, 1.0*BULLET_HEIGHT/2.0), Vec2::new(-1.0*BULLET_WIDTH/2.0, -1.0*BULLET_HEIGHT/2.0), Vec2::new(1.0*BULLET_WIDTH/2.0, -1.0*BULLET_HEIGHT/2.0), Vec2::new(1.0*BULLET_WIDTH/2.0, 1.0*BULLET_HEIGHT/2.0)];

struct TanksWorldData {
    camera: Camera2d,
    player_tank: Option<Tank>,
    enemy_tanks: WorldVec<Tank>,
    explosion_animation: Rc<Animation2d>,
    animation_list: WorldVec<AnimationObject2d>,
    bullets: WorldVec<Bullet>,
    world_borders: Vec<Polygon2d>,
    window_size: IVec2,
}

struct TanksRenderData {
    atlas_quad_mesh: Mesh,
    atlas_pipeline: wgpu::RenderPipeline,
    render_camera: RenderCamera,
    color_bind_group_layout: wgpu::BindGroupLayout,
    debug_lines_pipeline: wgpu::RenderPipeline,
    tanks_texture_atlas_bind_group: wgpu::BindGroup,
    tank_texture_atlas_util: TextureAtlasUtil,
    texture_atlas_util: TextureCoordUtil,
    atlas_instances: VecBuffer<InstanceAtlas>,
    debug_line_instances: Vec<InstanceDebugLine>,
    aspect_ratio_buffer: Buffer,
    aspect_ratio_bind_group: wgpu::BindGroup,
    font_ratio_buffer: Buffer,
    font_ratio_bind_group: wgpu::BindGroup,
    bitmap_font_pipeline: wgpu::RenderPipeline,
    open_sans_font_texture_atlas_bind_group: wgpu::BindGroup,
    open_sans_font_spec: FontSpecification,
    open_sans_font_instances: VecBuffer<CharacterInstance>,
    sdf_font_pipeline: wgpu::RenderPipeline,
    sdf_aem_distance_bind_group: wgpu::BindGroup,
    open_sans_sdf_font_texture_atlas_bind_group: wgpu::BindGroup,
    open_sans_sdf_font_spec: FontSpecification,
    open_sans_sdf_font_instances: VecBuffer<SdfCharacterInstance>,
}

struct Tank {
    transform2d: Transform2d,
    cannon_rotation: f32,
    collider: Polygon2d,
}

impl Tank {
    fn new(transform2d: Transform2d, cannon_rotation: f32) -> Self {
        let green_color = Vec3::new(0.0, 1.0, 0.0);
        Self {
            transform2d,
            cannon_rotation,
            collider: Polygon2d::new(TANK_COLLISION_POINTS[0].to_vec(), Vec2::splat(0.0), 0.0, green_color),
        }
    }
}

struct Bullet {
    transform2d: Transform2d,
    velocity: f32,
    collider: Polygon2d,
    bounces: u32,
    max_bounces: u32
}

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub fn run() {
    cfg_if::cfg_if!(
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Unable to initialize logger!");
        }
    );

    let mut app = App::<TanksWorldData, TanksRenderData>::new("tanks", startup, resize_event, update, prepare, render);
    app.run();
}

fn startup(input_manager: &mut InputManager, render_engine: &RenderEngine) -> (TanksWorldData, TanksRenderData) {
    input_manager.register_key_binding("up", KeyCode::KeyW);
    input_manager.register_key_binding("down", KeyCode::KeyS);
    input_manager.register_key_binding("left", KeyCode::KeyA);
    input_manager.register_key_binding("right", KeyCode::KeyD);
    input_manager.register_key_binding("shoot", KeyCode::Space);

    let camera_bind_group_layout = create_camera_bind_group_layout(&render_engine.device);
    let color_bind_group_layout = create_color_bind_group_layout(&render_engine.device);

    let texture_bind_group_layout = create_texture_bind_group_layout(&render_engine.device);
    let tanks_texture_atlas = Texture::from_bytes(
        &render_engine.device, 
        &render_engine.queue, 
        include_bytes_from_project_path!("/res/game_objects/tank_texture_atlas.png"), 
        "Tanks Texture Atlas"
    );

    let debug_lines_pipeline = create_debug_lines_pipeline2d(&render_engine.device, &render_engine.config, &camera_bind_group_layout, &color_bind_group_layout);

    let tank_texture_atlas_util = TextureAtlasUtil::new(128, 128, tanks_texture_atlas.width, tanks_texture_atlas.height);
    
    let mut explosion_animation = Animation2d::new();
    explosion_animation.add_animation_frame(AnimationFrame2d::new_from_seconds(0.75, tank_texture_atlas_util.get_texture_coords_from_atlas_coords(0, 4)));
    explosion_animation.add_animation_frame(AnimationFrame2d::new_from_seconds(0.75, tank_texture_atlas_util.get_texture_coords_from_atlas_coords(1, 4)));
    explosion_animation.add_animation_frame(AnimationFrame2d::new_from_seconds(0.75, tank_texture_atlas_util.get_texture_coords_from_atlas_coords(2, 4)));

    let aspect_ratio_bind_group_layout = create_aspect_ratio_bind_group_layout(&render_engine.device);
    let aspect_ratio_buffer = Buffer::create_buffer_during_init(&render_engine.device, String::from("Aspect Ratio Buffer"), bytemuck::bytes_of(&render_engine.aspect_ratio), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
    //TODO create my own bind group struct so no direct wgpu is required
    let aspect_ratio_bind_group = render_engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Aspect Ratio Bind Group"),
            layout: &aspect_ratio_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: aspect_ratio_buffer.as_entire_binding(),
                }
            ],
        });
    
    let font_ratio_bind_group_layout = create_font_ratio_bind_group_layout(&render_engine.device);
    let font_ratio = get_font_ratio(render_engine.window_width as f32);
    let font_ratio_buffer = Buffer::create_buffer_during_init(&render_engine.device, String::from("Font Ratio Buffer"), bytemuck::bytes_of(&font_ratio), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
    let font_ratio_bind_group = render_engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Font Ratio Bind Group"),
        layout: &font_ratio_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: font_ratio_buffer.as_entire_binding(),
            }
        ]
    });

    let (open_sans_font_spec,open_sans_font_texture_atlas) = get_font_from_defaults(DefaultFonts::OpenSans, &render_engine.device, &render_engine.queue);
    let mut open_sans_font_instances = VecBuffer::new(&render_engine.device, "OpenSans Instances", wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST);
    let mut text1 = construct_string_instance_data("Initial Test. INITIAL TEST", Vec2::new(-1.0, 0.5), 1.0, Vec3::new(0.0, 0.0, 0.0), false, &open_sans_font_spec);
    open_sans_font_instances.append(&mut text1);
    let mut text2 = construct_string_instance_data("Scales with the screen!", Vec2::new(0.0, -0.5), 1.0, Vec3::new(1.0, 1.0, 1.0), true, &open_sans_font_spec);
    open_sans_font_instances.append(&mut text2);
    open_sans_font_instances.update_buffer_data(&render_engine.device, &render_engine.queue);

    let sdf_aem_bind_group_layout = create_sdf_aemrange_bind_group_layout(&render_engine.device); 
    
    let (open_sans_sdf_font_spec, open_sans_sdf_font_texture_atlas) = get_font_from_defaults(DefaultFonts::OpenSansSDF, &render_engine.device, &render_engine.queue);
    let mut open_sans_sdf_font_instances = VecBuffer::new(&render_engine.device, "OpenSans SDF Instances", wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST);
    let mut text3 = construct_sdf_string_instance_data("Initial Test. INITIAL TEST", Vec2::new(0.0, 0.7), 1.0, Vec3::new(1.0, 1.0, 0.0), false, &open_sans_sdf_font_spec);
    open_sans_sdf_font_instances.append(&mut text3);
    let mut text4 = construct_sdf_string_instance_data("Scales with the screen!", Vec2::new(0.0, 0.5), 1.0, Vec3::new(0.0, 1.0, 1.0), true, &open_sans_sdf_font_spec);
    open_sans_sdf_font_instances.append(&mut text4);
    open_sans_sdf_font_instances.update_buffer_data(&render_engine.device, &render_engine.queue);

    let tanks_render_data = TanksRenderData { 
        atlas_quad_mesh: create_atlas_quad_mesh(&render_engine.device), 
        atlas_pipeline: create_atlas_pipeline2d(&render_engine.device, &render_engine.config, &camera_bind_group_layout, &texture_bind_group_layout), 
        color_bind_group_layout,
        debug_lines_pipeline,  
        tanks_texture_atlas_bind_group: create_texture_bind_group(
            &render_engine.device, 
            "Tanks Texture Atlas Bind Group", 
            &texture_bind_group_layout, 
            &tanks_texture_atlas.view, 
            &tanks_texture_atlas.sampler
        ),
        tank_texture_atlas_util,
        texture_atlas_util: TextureCoordUtil::new(tanks_texture_atlas.width, tanks_texture_atlas.height),
        render_camera: RenderCamera::new(&render_engine.device, &camera_bind_group_layout, "Primary Camera"),
        atlas_instances: VecBuffer::new(&render_engine.device, "Render Tank Instances", wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST), 
        debug_line_instances: Vec::new(),
        aspect_ratio_buffer,
        aspect_ratio_bind_group,
        font_ratio_buffer,
        font_ratio_bind_group,  
        bitmap_font_pipeline: create_bitmap_font_pipeline(&render_engine.device, &render_engine.config, &texture_bind_group_layout, &aspect_ratio_bind_group_layout, &font_ratio_bind_group_layout),
        open_sans_font_texture_atlas_bind_group: create_texture_bind_group(
            &render_engine.device, 
            "Open Sans Texture Atlas Bind Group", 
            &texture_bind_group_layout, 
            &open_sans_font_texture_atlas.view, 
            &open_sans_font_texture_atlas.sampler),
        open_sans_font_spec,
        open_sans_font_instances,
        sdf_font_pipeline: create_sdf_font_pipeline(&render_engine.device, &render_engine.config, &texture_bind_group_layout, &aspect_ratio_bind_group_layout, &font_ratio_bind_group_layout, &sdf_aem_bind_group_layout),
        sdf_aem_distance_bind_group: get_sdf_aem_distance_bind_group(&open_sans_sdf_font_spec, &sdf_aem_bind_group_layout, &render_engine.device),
        open_sans_sdf_font_spec,
        open_sans_sdf_font_texture_atlas_bind_group: create_texture_bind_group(
            &render_engine.device, 
            "Open Sans SDF Texture Atlas Bind Group", 
            &texture_bind_group_layout, 
            &open_sans_sdf_font_texture_atlas.view, 
            &open_sans_sdf_font_texture_atlas.sampler),
        open_sans_sdf_font_instances
    };

    let green_color = Vec3::new(0.0, 1.0, 0.0);

    let mut enemy_tanks = WorldVec::<Tank>::new();
    enemy_tanks.push(Tank::new(
        Transform2d::new(Vec2::new(2.0, -2.0), 35.0, Vec2::splat(1.0)), 
        45.0
    ));
    enemy_tanks.push(Tank::new(
        Transform2d::new(Vec2::new(-2.0, 2.0), 72.0, Vec2::splat(1.0)), 
        232.0)
    );

    let tanks_world_data = TanksWorldData { 
        camera: Camera2d::new(
            Vec3::new(0.0, 0.0, -1.0),
            CameraProjection::Orthographic(OrthographicProjection {
                width: 10.0,
                height: 10.0,
                near_clip: 0.1,
                far_clip: 1000.0,
                aspect_ratio: render_engine.aspect_ratio,
            })
        ),
        player_tank: Some(Tank::new(
            Transform2d::new(Vec2::splat(0.0), 0.0, Vec2::splat(1.0)), 
            0.0)
        ), 
        enemy_tanks,
        explosion_animation: Rc::new(explosion_animation),
        animation_list: WorldVec::new(), 
        bullets: WorldVec::new(),
        world_borders: vec![
            Polygon2d::new(vec![Vec2::new(-8.5, 4.5), Vec2::new(-8.5, -4.5), Vec2::new(-8.0, -4.5), Vec2::new(-8.0, 4.5)], Vec2::splat(0.0), 0.0, green_color),
            Polygon2d::new(vec![Vec2::new(8.0, 4.5), Vec2::new(8.0, -4.5), Vec2::new(8.5, -4.5), Vec2::new(8.5, 4.5)], Vec2::splat(0.0), 0.0, green_color),
            Polygon2d::new(vec![Vec2::new(-8.0, 5.0), Vec2::new(-8.0, 4.5), Vec2::new(8.0, 4.5), Vec2::new(8.0, 5.0)], Vec2::splat(0.0), 0.0, green_color),
            Polygon2d::new(vec![Vec2::new(-8.0, -4.5), Vec2::new(-8.0, -5.0), Vec2::new(8.0, -5.0), Vec2::new(8.0, -4.5)], Vec2::splat(0.0), 0.0, green_color)
        ],
        window_size: IVec2::new(render_engine.window_width as i32, render_engine.window_height as i32) 
    };

    (tanks_world_data, tanks_render_data)
}

fn update(tanks_world_data: &mut TanksWorldData, input_manager: &InputManager) {
    update_player_tank_controller(tanks_world_data, input_manager);
    update_bullet_movement(tanks_world_data);
    update_collision_shapes(tanks_world_data);
    update_collision_tank_bullet(tanks_world_data);
    update_world_border_collisions(tanks_world_data);
    update_explosion_animations(tanks_world_data);
}

fn update_player_tank_controller(tanks_world_data: &mut TanksWorldData, input_manager: &InputManager) {
    if let Some(player_tank) = &mut tanks_world_data.player_tank {
        let tank_transform2d = &mut player_tank.transform2d;

        let rotation_speed = 0.5;
        let movement_speed = 0.005;

        let left_key_state = input_manager.get_key_state("left").unwrap();
        let right_key_state = input_manager.get_key_state("right").unwrap();
        let up_key_state = input_manager.get_key_state("up").unwrap();
        let down_key_state = input_manager.get_key_state("down").unwrap();
        let shoot_key_state = input_manager.get_key_state("shoot").unwrap();

        let mut movement_direction = IVec2::new(0, 0);
        if left_key_state.get_is_down() {
            movement_direction.x -= 1;
        }
        if right_key_state.get_is_down() {
            movement_direction.x += 1;
        }
        if up_key_state.get_is_down() {
            movement_direction.y += 1;
        }
        if down_key_state.get_is_down() {
            movement_direction.y -= 1;
        }

        if movement_direction.x != 0 || movement_direction.y != 0 {
            let target_degrees = f32::atan2(movement_direction.y as f32, movement_direction.x as f32).to_degrees();
            let target_degrees_constrained = (target_degrees + 360.0) % 360.0;

            let current_rotation = tank_transform2d.get_rotation();

            let mut rotation_diff_degrees = target_degrees_constrained - current_rotation;
            let rotation_diff_degrees_abs = f32::abs(rotation_diff_degrees);
            if rotation_diff_degrees_abs > 180.0 {rotation_diff_degrees *= -1.0; }

            if rotation_diff_degrees_abs < 0.0001 {
                tank_transform2d.set_rotation(target_degrees_constrained);
            }
            else {
                tank_transform2d.rotate(rotation_diff_degrees.signum() * rotation_speed);
            }

            let movement_vector = Vec2::new(tank_transform2d.get_rotation().to_radians().cos(), tank_transform2d.get_rotation().to_radians().sin());
            tank_transform2d.translate(movement_vector * movement_speed); 
        }

        let pixel_mouse_coords = input_manager.get_mouse_physical();
        let mouse_coords = tanks_world_data.camera.convert_screen_space_to_world_space(pixel_mouse_coords, tanks_world_data.window_size);
        let cannon_center = player_tank.transform2d.get_translation() + Vec2::new(0.0, 0.01875);
        let tank_mouse_difference = mouse_coords.xy() - cannon_center;
        player_tank.cannon_rotation = (f32::atan2(tank_mouse_difference.y, tank_mouse_difference.x).to_degrees() + 360.0) % 360.0;

        if shoot_key_state.get_was_pressed() {
            let offset_vector = Vec2::new(player_tank.cannon_rotation.to_radians().cos(), player_tank.cannon_rotation.to_radians().sin());
            let translation = player_tank.transform2d.get_translation() + (offset_vector * 0.65);
            let bullet = Bullet {
                transform2d: Transform2d::new(translation, player_tank.cannon_rotation, Vec2::splat(1.0)),
                velocity: 0.01,
                collider: Polygon2d::new(BULLET_COLLISION_POINTS.to_vec(), translation, player_tank.cannon_rotation, Vec3::new(0.0, 1.0, 0.0)),
                bounces: 0,
                max_bounces: 4,
            };
            tanks_world_data.bullets.push(bullet);
        }
    }
}

fn update_bullet_movement(tanks_world_data: &mut TanksWorldData) {
    for bullet in &mut tanks_world_data.bullets {
        let movement_vector = Vec2::new(bullet.transform2d.get_rotation().to_radians().cos(), bullet.transform2d.get_rotation().to_radians().sin());
        let bullet_velocity = bullet.velocity;
        bullet.transform2d.translate(movement_vector * bullet_velocity);
    }
}

fn update_collision_shapes(tanks_world_data: &mut TanksWorldData) {
    let mut tanks = Vec::<&mut Tank>::new();
    for enemy_tank in &mut tanks_world_data.enemy_tanks {
        tanks.push(enemy_tank);
    }
    if let Some(player_tank) = &mut tanks_world_data.player_tank {
        tanks.push(player_tank);
    }
    for tank in tanks {
        let atlas_index = get_tank_atlas_index(tank.transform2d.get_rotation());
        let debug_color = tank.collider.debug_color;
        tank.collider = Polygon2d::new(TANK_COLLISION_POINTS[((atlas_index+4)%8) as usize].to_vec(), tank.transform2d.get_translation(), 0.0, debug_color);
    }

    for bullet in &mut tanks_world_data.bullets {
        let bullet_translation = bullet.transform2d.get_translation();
        let bullet_rotation = bullet.transform2d.get_rotation();
        bullet.collider.set_translation_rotation(bullet_translation, bullet_rotation);
    }
}

fn update_world_border_collisions(tanks_world_data: &mut TanksWorldData) {
    let mut tanks = Vec::<&mut Tank>::new();
    for enemy_tank in &mut tanks_world_data.enemy_tanks {
        tanks.push(enemy_tank);
    }
    if let Some(player_tank) = &mut tanks_world_data.player_tank {
        tanks.push(player_tank);
    }
    for tank in tanks {
        for world_border in &tanks_world_data.world_borders {
            let intersection_result = tank.collider.get_intersection_result(&world_border);
            if intersection_result.collision {
                tank.transform2d.translate(intersection_result.normal * intersection_result.penetration_val);
            }
        }
    }

    let mut remove_list = Vec::new();
    for bullet in &mut tanks_world_data.bullets {
        for world_border in &tanks_world_data.world_borders {
            let intersection_result = bullet.collider.get_intersection_result(&world_border);
            if intersection_result.collision {
                if bullet.bounces == bullet.max_bounces {
                    remove_list.push(bullet.get_index_handle());
                }
                else {
                    let direction_vector = Vec2::new(bullet.transform2d.get_rotation().to_radians().cos(), bullet.transform2d.get_rotation().to_radians().sin());
                    let bounced_direction_vector = direction_vector - 2.0 * direction_vector.dot(intersection_result.normal) * intersection_result.normal;
                    bullet.transform2d.set_rotation(f32::atan2(bounced_direction_vector.y, bounced_direction_vector.x).to_degrees());
                    bullet.bounces += 1;
                }
            }
        }
    }

    for bullet_index_handle in remove_list {
        let index_option = bullet_index_handle.borrow().clone();
        tanks_world_data.bullets.swap_remove(index_option);
    }
    
}

fn update_collision_tank_bullet(tanks_world_data: &mut TanksWorldData) {
    let mut tank_remove_list = Vec::new();
    let mut bullet_remove_list = Vec::new();

    if !tanks_world_data.bullets.is_empty() {
        for bullet_1_index in 0..tanks_world_data.bullets.len() - 1 {
            for bullet_2_index in bullet_1_index + 1..tanks_world_data.bullets.len() {
                if bullet_1_index != bullet_2_index {
                    let bullet_1 = &tanks_world_data.bullets[bullet_1_index];
                    let bullet_2 = &tanks_world_data.bullets[bullet_2_index];
                    let intersection_result = bullet_1.collider.get_intersection_result(&bullet_2.collider);
                    if intersection_result.collision {
                        bullet_remove_list.push(bullet_1.get_index_handle());
                        bullet_remove_list.push(bullet_2.get_index_handle());
                    }
                }
            }
        }
    }

    for tank in &tanks_world_data.enemy_tanks {
        for bullet in &tanks_world_data.bullets {
            let intersection_result = tank.collider.get_intersection_result(&bullet.collider);
            if intersection_result.collision {
                tank_remove_list.push(tank.get_index_handle());
                bullet_remove_list.push(bullet.get_index_handle());
            }
        }
    }

    let mut player_bullet_collision = false;
    if let Some(player_tank) = &tanks_world_data.player_tank {
        for bullet in &tanks_world_data.bullets {
            let intersection_result = player_tank.collider.get_intersection_result(&bullet.collider);
            if intersection_result.collision {
                player_bullet_collision = true;
                bullet_remove_list.push(bullet.get_index_handle());
            }
        }
    }

    if player_bullet_collision {
        let explosion_transform = tanks_world_data.player_tank.as_ref().unwrap().transform2d.clone();
        let explosion_animation_object = AnimationObject2d::new(explosion_transform, Rc::clone(&tanks_world_data.explosion_animation));
        tanks_world_data.animation_list.push(explosion_animation_object);
        tanks_world_data.player_tank = None;
    }

    for tank_index_handle in tank_remove_list {
        let index_option = tank_index_handle.borrow().clone();
        let removed_tank_option = tanks_world_data.enemy_tanks.swap_remove(index_option);
        if let Some(removed_tank) = removed_tank_option {
            let explosion_transform = removed_tank.transform2d.clone();
            let explosion_animation_object = AnimationObject2d::new(explosion_transform, Rc::clone(&tanks_world_data.explosion_animation));
            tanks_world_data.animation_list.push(explosion_animation_object);
        }
    }

    for bullet_index_handle in bullet_remove_list {
        let index_option = bullet_index_handle.borrow().clone();
        tanks_world_data.bullets.swap_remove(index_option);
    }
}

fn update_explosion_animations(tanks_world_data: &mut TanksWorldData) {
    let mut explosion_remove_list = Vec::new();
    for explosion_animation in &mut tanks_world_data.animation_list {
        if explosion_animation.is_animation_complete() {
            explosion_remove_list.push(explosion_animation.get_index_handle());
        }
        else {
            explosion_animation.run_animation();
        }
    }

    for index_handle in explosion_remove_list {
        let index_option = index_handle.borrow().clone();
        tanks_world_data.animation_list.swap_remove(index_option);
    }
}

fn resize_event(tanks_world_data: &mut TanksWorldData, tanks_render_data: &mut TanksRenderData, render_engine: &RenderEngine) {
    tanks_world_data.window_size = IVec2::new(render_engine.window_width as i32, render_engine.window_height as i32);
    tanks_world_data.camera.update_aspect_ratio(render_engine.aspect_ratio);
    tanks_render_data.aspect_ratio_buffer.update_buffer_data(&render_engine.device, &render_engine.queue, bytemuck::bytes_of(&render_engine.aspect_ratio));
    
    let font_ratio = get_font_ratio(render_engine.window_width as f32);
    tanks_render_data.font_ratio_buffer.update_buffer_data(&render_engine.device, &render_engine.queue, bytemuck::bytes_of(&font_ratio));
}

fn prepare(tanks_world_data: &mut TanksWorldData, tanks_render_data: &mut TanksRenderData, render_engine: &RenderEngine) {
    let camera_matrix = &tanks_world_data.camera.view_projection_matrix;
    let render_camera = &mut tanks_render_data.render_camera;
    render_camera.update_uniform_buffer(
        camera_matrix.clone(), 
        &render_engine.queue
    );
    
    prepare_atlas_objects(tanks_world_data, tanks_render_data, render_engine);
    prepare_debug_lines(tanks_world_data, tanks_render_data, render_engine);
}

fn prepare_atlas_objects(tanks_world_data: &mut TanksWorldData, tanks_render_data: &mut TanksRenderData, render_engine: &RenderEngine) {
    tanks_render_data.atlas_instances.clear();

    prepare_tanks(tanks_world_data, tanks_render_data);
    prepare_bullets(tanks_world_data, tanks_render_data);
    prepare_animations(tanks_world_data, tanks_render_data);

    tanks_render_data.atlas_instances.update_buffer_data(&render_engine.device, &render_engine.queue);
}

fn prepare_tanks(tanks_world_data: &mut TanksWorldData, tanks_render_data: &mut TanksRenderData) {
    let mut tanks = Vec::<&Tank>::new();
    for enemy_tank in &tanks_world_data.enemy_tanks {
        tanks.push(enemy_tank);
    }
    if let Some(player_tank) = &tanks_world_data.player_tank {
        tanks.push(player_tank);
    }

    let sprite_size = tanks_render_data.tank_texture_atlas_util.get_atlas_size_in_tex_coords();

    for tank in tanks {
        let base_atlas_index = get_tank_atlas_index(tank.transform2d.get_rotation());
        let base_atlas_index_x = base_atlas_index % 16;
        let base_atlas_index_y = base_atlas_index / 16;
        let base_texture_coords = tanks_render_data.tank_texture_atlas_util.get_texture_coords_from_atlas_coords(base_atlas_index_x, base_atlas_index_y);
        let tank_base_instance = InstanceAtlas {
            model: Mat4::from_translation(Vec3::new(tank.transform2d.get_translation().x, tank.transform2d.get_translation().y, 0.0)),
            tex_coords: base_texture_coords,
            sprite_size: sprite_size
        };
        tanks_render_data.atlas_instances.push(tank_base_instance);

        let cannon_atlas_index = get_tank_cannon_atlas_index(tank.cannon_rotation);
        let cannon_atlas_index_x = cannon_atlas_index % 16;
        let cannon_atlas_index_y = cannon_atlas_index / 16;
        let cannon_texture_coords = tanks_render_data.tank_texture_atlas_util.get_texture_coords_from_atlas_coords(cannon_atlas_index_x, cannon_atlas_index_y);
        let tank_cannon_instance = InstanceAtlas {
            model: Mat4::from_translation(Vec3::new(tank.transform2d.get_translation().x, tank.transform2d.get_translation().y, 0.1)),
            tex_coords: cannon_texture_coords,
            sprite_size: sprite_size
        };
        tanks_render_data.atlas_instances.push(tank_cannon_instance);
    }
}

fn prepare_bullets(tanks_world_data: &mut TanksWorldData, tanks_render_data: &mut TanksRenderData) {
    for bullet in &tanks_world_data.bullets {
        let bullet_texture_coords = tanks_render_data.texture_atlas_util.get_texture_coord_from_pixels(47, 442);
        let sprite_size = tanks_render_data.texture_atlas_util.get_texture_coord_from_pixels(34, 12);
        let scale = Vec3::new(BULLET_WIDTH, BULLET_HEIGHT, 0.0);
        let rotation = Quat::from_rotation_z(bullet.transform2d.get_rotation().to_radians());
        let translation = Vec3::new(bullet.transform2d.get_translation().x, bullet.transform2d.get_translation().y, 0.1);

        let bullet_instance = InstanceAtlas {
            model: Mat4::from_scale_rotation_translation(scale, rotation, translation),
            tex_coords: bullet_texture_coords,
            sprite_size: sprite_size,
        };
        tanks_render_data.atlas_instances.push(bullet_instance);
    }
}

fn prepare_animations(tanks_world_data: &mut TanksWorldData, tanks_render_data: &mut TanksRenderData) {
    for animation in &tanks_world_data.animation_list {
        let sprite_size = tanks_render_data.tank_texture_atlas_util.get_atlas_size_in_tex_coords();
        let animation_texture_coords = animation.get_texture_coords_for_current_frame();
        let animation_instance = InstanceAtlas {
            model: Mat4::from_translation(Vec3::new(animation.transform2d.get_translation().x, animation.transform2d.get_translation().y, 0.1)),
            tex_coords: animation_texture_coords,
            sprite_size: sprite_size
        };

        tanks_render_data.atlas_instances.push(animation_instance);
    }
}

fn prepare_debug_lines(tanks_world_data: &mut TanksWorldData, tanks_render_data: &mut TanksRenderData, render_engine: &RenderEngine) {
    tanks_render_data.debug_line_instances.clear();
    let mut polygon2ds = Vec::new();
    if let Some(player_tank) = &mut tanks_world_data.player_tank {
        polygon2ds.push(&player_tank.collider);
    }
    for tank in &tanks_world_data.enemy_tanks {
        polygon2ds.push(&tank.collider);
    }
    for bullet in &tanks_world_data.bullets {
        polygon2ds.push(&bullet.collider);
    }
    polygon2ds.extend(tanks_world_data.world_borders.iter());
    for polygon2d in polygon2ds {
        let vertices: Vec<Vec3> = polygon2d.points.clone().into_iter().map(|vec2| vec2.xy().extend(10.0)).collect();
        let vertex_buffer = Buffer::create_buffer_during_init(
            &render_engine.device, 
            String::from("Debug Line Vertex Buffer"), 
            bytemuck::cast_slice(&vertices), 
            wgpu::BufferUsages::VERTEX
        );
        let mut index_list: Vec<u16> = (0..polygon2d.points.len() as u16).collect();
        index_list.push(0);
        let index_buffer = Buffer::create_buffer_during_init(
            &render_engine.device, 
            String::from("Debug Line Index Buffer"), 
            bytemuck::cast_slice(&index_list), 
            BufferUsages::INDEX
        );

        let num_indices = polygon2d.points.len() + 1;

        let mesh = Mesh {
            name: String::from("Debug Line"),
            vertex_buffer: vertex_buffer,
            index_buffer: index_buffer,
            num_elements: num_indices as u32,
        };

        let color = polygon2d.debug_color.xyz().extend(1.0);

        let color_uniform = Buffer::create_buffer_during_init(
            &render_engine.device, 
            String::from("Color Uniform"),
            bytemuck::cast_slice(&color.to_array()), 
            BufferUsages::UNIFORM
        );

        //TODO create my own bind group struct so no wgpu interfacing needs to occur
        let color_bind_group = render_engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Debug Line Bind Group"),
            layout: &tanks_render_data.color_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: color_uniform.as_entire_binding(),
                }
            ],
        });

        tanks_render_data.debug_line_instances.push(InstanceDebugLine { mesh, color_uniform, color_bind_group });
    }
}

fn render(tanks_render_data: &TanksRenderData, render_engine: &RenderEngine) -> Result<(), wgpu::SurfaceError> {
    let mut encoder = create_command_encoder(&render_engine.device, "Command Encoder");
    let (mut primary_render_pass, output) = render_engine.render_frame_setup(&mut encoder)?;

    render_atlas_objects(tanks_render_data, &mut primary_render_pass);
    render_debug_lines(tanks_render_data, &mut primary_render_pass);
    render_opensans_font(tanks_render_data, &mut primary_render_pass);
    render_opensans_sdf_font(tanks_render_data, &mut primary_render_pass);

    drop(primary_render_pass);
    render_engine.render_frame_present(output, encoder);
    Ok(())
}

fn render_atlas_objects<'a: 'b, 'b>(tanks_render_data: &'a TanksRenderData, primary_render_pass: &'b mut wgpu::RenderPass<'a>) {
    primary_render_pass.set_pipeline(&tanks_render_data.atlas_pipeline);
    primary_render_pass.set_vertex_buffer(1, tanks_render_data.atlas_instances.get_buffer().slice(..));
    primary_render_pass.set_bind_group(1, &tanks_render_data.tanks_texture_atlas_bind_group, &[]);
    primary_render_pass.draw_mesh_instanced(&tanks_render_data.atlas_quad_mesh, 0..tanks_render_data.atlas_instances.len() as u32, &tanks_render_data.render_camera.camera_bind_group);
}

fn render_debug_lines<'a: 'b, 'b>(tanks_render_data: &'a TanksRenderData, primary_render_pass: &'b mut wgpu::RenderPass<'a>) {
    primary_render_pass.set_pipeline(&tanks_render_data.debug_lines_pipeline);

    for debug_lines in &tanks_render_data.debug_line_instances {
        primary_render_pass.set_bind_group(1, &debug_lines.color_bind_group, &[]);
        primary_render_pass.draw_mesh(&debug_lines.mesh, &tanks_render_data.render_camera.camera_bind_group);
    }
}

fn render_opensans_font<'a: 'b, 'b>(tanks_render_data: &'a TanksRenderData, primary_render_pass: &'b mut wgpu::RenderPass<'a>) {
    primary_render_pass.set_pipeline(&tanks_render_data.bitmap_font_pipeline);
    primary_render_pass.set_vertex_buffer(1, tanks_render_data.open_sans_font_instances.get_buffer().slice(..));
    primary_render_pass.set_bind_group(0, &tanks_render_data.open_sans_font_texture_atlas_bind_group, &[]);
    primary_render_pass.set_bind_group(1, &tanks_render_data.aspect_ratio_bind_group, &[]);
    primary_render_pass.set_bind_group(2, &tanks_render_data.font_ratio_bind_group, &[]);
    primary_render_pass.draw_mesh_instanced_no_camera(&tanks_render_data.atlas_quad_mesh, 0..tanks_render_data.open_sans_font_instances.len() as u32);
}

fn render_opensans_sdf_font<'a: 'b, 'b>(tanks_render_data: &'a TanksRenderData, primary_render_pass: &'b mut wgpu::RenderPass<'a>) {
    primary_render_pass.set_pipeline(&tanks_render_data.sdf_font_pipeline);
    primary_render_pass.set_vertex_buffer(1, tanks_render_data.open_sans_sdf_font_instances.get_buffer().slice(..));
    primary_render_pass.set_bind_group(0, &tanks_render_data.open_sans_sdf_font_texture_atlas_bind_group, &[]);
    primary_render_pass.set_bind_group(1, &tanks_render_data.aspect_ratio_bind_group, &[]);
    primary_render_pass.set_bind_group(2, &tanks_render_data.font_ratio_bind_group, &[]);
    primary_render_pass.set_bind_group(3, &tanks_render_data.sdf_aem_distance_bind_group, &[]);
    primary_render_pass.draw_mesh_instanced_no_camera(&tanks_render_data.atlas_quad_mesh, 0..tanks_render_data.open_sans_sdf_font_instances.len() as u32);
}

fn get_tank_atlas_index(rotation: f32) -> u32 {
    let step = 360.0 / 16.0;
    let half_step = step / 2.0;
    let adjusted_rotation = ((rotation + half_step) + 360.0) % 360.0;
    let index = (adjusted_rotation / step) as u32;
    (index + 4) % 8
}

fn get_tank_cannon_atlas_index(rotation: f32) -> u32 {
    let step = 360.0 / 16.0;
    let half_step = step / 2.0;
    let adjusted_rotation = ((rotation + half_step) + 360.0) % 360.0;
    let index = (adjusted_rotation / step) as u32;
    index + 16
}