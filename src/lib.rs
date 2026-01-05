use xenofrost::core::app::App;
use xenofrost::core::input_manager::{InputManager, KeyCode};
use xenofrost::core::math::bounding2d::Polygon2d;
use xenofrost::core::render_engine::buffer::{Buffer, VecBuffer};
use xenofrost::core::render_engine::mesh::{Mesh, create_atlas_quad_mesh};
use xenofrost::core::render_engine::pipeline::{InstanceAtlas, InstanceDebugLine, create_atlas_pipeline2d, create_color_bind_group_layout, create_debug_lines_pipeline2d};
use xenofrost::core::render_engine::render_camera::{RenderCamera, create_camera_bind_group_layout};
use xenofrost::core::render_engine::texture::{Texture, create_texture_bind_group, create_texture_bind_group_layout};
use xenofrost::core::render_engine::wgpu::BufferUsages;
use xenofrost::core::render_engine::{DrawMesh, bytemuck, wgpu};
use xenofrost::core::render_engine::{RenderEngine, create_command_encoder};
use xenofrost::core::math::{IVec2, Mat4, Transform2d, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles};
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

const BULLET_COLLISION_POINTS: &[Vec2] = &[Vec2::new(-0.1, 0.1), Vec2::new(-0.1, -0.1), Vec2::new(0.1, -0.1), Vec2::new(0.1, 0.1)];

struct TanksWorldData {
    camera: Camera2d,
    player_tank: Tank,
    enemy_tanks: Vec<Tank>,
    bullets: Vec<Bullet>,
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
    render_tank_instances: VecBuffer<InstanceAtlas>,
    debug_line_instances: Vec<InstanceDebugLine>
}

struct Tank {
    transform2d: Transform2d,
    cannon_rotation: f32,
    collider: Polygon2d
}

struct Bullet {
    transform2d: Transform2d,
    velocity: f32,
    damage: f32,
    collider: Polygon2d
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
        render_camera: RenderCamera::new(&render_engine.device, &camera_bind_group_layout, "Primary Camera"),
        render_tank_instances: VecBuffer::new(&render_engine.device, "Render Tank Instances", wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST), 
        debug_line_instances: Vec::new()
    };

    let green_color = Vec3::new(0.0, 1.0, 0.0);

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
        player_tank: Tank { 
            transform2d: Transform2d::new(Vec2::splat(0.0), 0.0, Vec2::splat(1.0)), 
            cannon_rotation: 0.0,
            collider: Polygon2d::new(TANK_COLLISION_POINTS[0].to_vec(), Vec2::splat(0.0), 0.0, green_color),
        }, 
        enemy_tanks: vec![
            Tank {
                transform2d: Transform2d::new(Vec2::new(2.0, -2.0), 35.0, Vec2::splat(1.0)), 
                cannon_rotation: 45.0,
                collider: Polygon2d::new(TANK_COLLISION_POINTS[0].to_vec(), Vec2::splat(0.0), 0.0, green_color),
            },
            Tank {
                transform2d: Transform2d::new(Vec2::new(-2.0, 2.0), 72.0, Vec2::splat(1.0)), 
                cannon_rotation: 232.0,
                collider: Polygon2d::new(TANK_COLLISION_POINTS[0].to_vec(), Vec2::splat(0.0), 0.0, green_color),
            }, 
        ], 
        bullets: Vec::new(),
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
    update_world_border_collisions(tanks_world_data);
    update_collision_debug_color_intersection(tanks_world_data);
}

fn update_player_tank_controller(tanks_world_data: &mut TanksWorldData, input_manager: &InputManager) {
    let tank_transform2d = &mut tanks_world_data.player_tank.transform2d;

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
    let cannon_center = tanks_world_data.player_tank.transform2d.get_translation() + Vec2::new(0.0, 0.01875);
    let tank_mouse_difference = mouse_coords.xy() - cannon_center;
    tanks_world_data.player_tank.cannon_rotation = (f32::atan2(tank_mouse_difference.y, tank_mouse_difference.x).to_degrees() + 360.0) % 360.0;

    if shoot_key_state.get_was_pressed() {
        let bullet = Bullet {
            transform2d: Transform2d::new(tanks_world_data.player_tank.transform2d.get_translation(), tanks_world_data.player_tank.cannon_rotation, Vec2::splat(1.0)),
            velocity: 0.01,
            damage: 1.0,
            collider: Polygon2d::new(BULLET_COLLISION_POINTS.to_vec(), tanks_world_data.player_tank.transform2d.get_translation(), tanks_world_data.player_tank.cannon_rotation, Vec3::new(0.0, 1.0, 0.0))
        };
        tanks_world_data.bullets.push(bullet);
    }
}

fn update_bullet_movement(tanks_world_data: &mut TanksWorldData) {
    for bullet in &mut tanks_world_data.bullets {
        let movement_vector = Vec2::new(bullet.transform2d.get_rotation().to_radians().cos(), bullet.transform2d.get_rotation().to_radians().sin());
        bullet.transform2d.translate(movement_vector * bullet.velocity);
    }
}

fn update_collision_shapes(tanks_world_data: &mut TanksWorldData) {
    let mut tanks = vec![&mut tanks_world_data.player_tank];
    tanks.extend(tanks_world_data.enemy_tanks.iter_mut());
    for tank in tanks {
        let atlas_index = get_tank_atlas_index(tank.transform2d.get_rotation());
        let debug_color = tank.collider.debug_color;
        tank.collider = Polygon2d::new(TANK_COLLISION_POINTS[((atlas_index+4)%8) as usize].to_vec(), tank.transform2d.get_translation(), 0.0, debug_color);
    }

    for bullet in &mut tanks_world_data.bullets {
        bullet.collider.set_translation_rotation(bullet.transform2d.get_translation(), bullet.transform2d.get_rotation());
    }
}

fn update_world_border_collisions(tanks_world_data: &mut TanksWorldData) {
    let mut tanks = vec![&mut tanks_world_data.player_tank];
    tanks.extend(tanks_world_data.enemy_tanks.iter_mut());
    for tank in tanks {
        for world_border in &tanks_world_data.world_borders {
            let intersection_result = tank.collider.get_intersection_result(&world_border);
            if intersection_result.collision {
                tank.transform2d.translate(intersection_result.normal * intersection_result.penetration_val);
            }
        }
    }

    for bullet in &mut tanks_world_data.bullets {
        for world_border in &tanks_world_data.world_borders {
            let intersection_result = bullet.collider.get_intersection_result(&world_border);
            if intersection_result.collision {
                bullet.transform2d.translate(intersection_result.normal * intersection_result.penetration_val);
            }
        }
    }
    
}

fn update_collision_debug_color_intersection(tanks_world_data: &mut TanksWorldData) {
    let mut colliders = vec![&mut tanks_world_data.player_tank.collider];
    for tank in &mut tanks_world_data.enemy_tanks {
        colliders.push(&mut tank.collider);
    }
    for bullet in &mut tanks_world_data.bullets {
        colliders.push(&mut bullet.collider);
    }
    colliders.extend(tanks_world_data.world_borders.iter_mut());

    for collider_index in 0..colliders.len() {
        colliders[collider_index].debug_color = Vec3::new(0.0, 1.0, 0.0);

        for other_collider_index in 0..colliders.len() {
            if collider_index == other_collider_index {
                continue;
            }
            
            let intersection_result = colliders[collider_index].get_intersection_result(colliders[other_collider_index]);
            if intersection_result.collision {
                colliders[collider_index].debug_color = Vec3::new(1.0, 0.0, 0.0);
            }
        }
    }
}

fn resize_event(tanks_world_data: &mut TanksWorldData, _tanks_render_data: &mut TanksRenderData, render_engine: &RenderEngine) {
    tanks_world_data.window_size = IVec2::new(render_engine.window_width as i32, render_engine.window_height as i32);
    tanks_world_data.camera.update_aspect_ratio(render_engine.aspect_ratio);
}

fn prepare(tanks_world_data: &mut TanksWorldData, tanks_render_data: &mut TanksRenderData, render_engine: &RenderEngine) {
    let camera_matrix = &tanks_world_data.camera.view_projection_matrix;
    let render_camera = &mut tanks_render_data.render_camera;
    render_camera.update_uniform_buffer(
        camera_matrix.clone(), 
        &render_engine.queue
    );
    
    prepare_tanks(tanks_world_data, tanks_render_data, render_engine);
    prepare_debug_lines(tanks_world_data, tanks_render_data, render_engine);
}

fn prepare_tanks(tanks_world_data: &mut TanksWorldData, tanks_render_data: &mut TanksRenderData, render_engine: &RenderEngine) {
    tanks_render_data.render_tank_instances.clear();
    let mut tanks = vec![&tanks_world_data.player_tank];
    tanks.extend(tanks_world_data.enemy_tanks.iter());
    for tank in tanks {
        let base_atlas_index = get_tank_atlas_index(tank.transform2d.get_rotation());
        let base_atlas_tex_coords_x = base_atlas_index % 16;
        let base_atlas_tex_coords_y = base_atlas_index / 16;
        let tank_base_instance = InstanceAtlas {
            model: Mat4::from_translation(Vec3::new(tank.transform2d.get_translation().x, tank.transform2d.get_translation().y, 0.0)),
            tex_coords: Vec2::new(0.0625*base_atlas_tex_coords_x as f32, 0.0625*base_atlas_tex_coords_y as f32),
            sprite_size: Vec2::new(0.0625, 0.0625)
        };
        tanks_render_data.render_tank_instances.push(tank_base_instance);

        let cannon_atlas_index = get_tank_cannon_atlas_index(tank.cannon_rotation);
        let cannon_atlas_tex_coords_x = cannon_atlas_index % 16;
        let cannon_atlas_tex_coords_y = cannon_atlas_index / 16;
        let tank_cannon_instance = InstanceAtlas {
            model: Mat4::from_translation(Vec3::new(tank.transform2d.get_translation().x, tank.transform2d.get_translation().y, 0.1)),
            tex_coords: Vec2::new(0.0625*cannon_atlas_tex_coords_x as f32, 0.0625*cannon_atlas_tex_coords_y as f32),
            sprite_size: Vec2::new(0.0625, 0.0625)
        };
        tanks_render_data.render_tank_instances.push(tank_cannon_instance);
        tanks_render_data.render_tank_instances.update_buffer_data(&render_engine.device, &render_engine.queue);
    }
}

fn prepare_debug_lines(tanks_world_data: &mut TanksWorldData, tanks_render_data: &mut TanksRenderData, render_engine: &RenderEngine) {
    tanks_render_data.debug_line_instances.clear();
    let mut polygon2ds = vec![&tanks_world_data.player_tank.collider];
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

    render_tanks(tanks_render_data, &mut primary_render_pass);
    render_debug_lines(tanks_render_data, &mut primary_render_pass);

    drop(primary_render_pass);
    render_engine.render_frame_present(output, encoder);
    Ok(())
}

fn render_tanks<'a: 'b, 'b>(tanks_render_data: &'a TanksRenderData, primary_render_pass: &'b mut wgpu::RenderPass<'a>) {
    primary_render_pass.set_pipeline(&tanks_render_data.atlas_pipeline);
    primary_render_pass.set_vertex_buffer(1, tanks_render_data.render_tank_instances.get_buffer().slice(..));
    primary_render_pass.set_bind_group(1, &tanks_render_data.tanks_texture_atlas_bind_group, &[]);
    primary_render_pass.draw_mesh_instanced(&tanks_render_data.atlas_quad_mesh, 0..tanks_render_data.render_tank_instances.len() as u32, &tanks_render_data.render_camera.camera_bind_group);
}

fn render_debug_lines<'a: 'b, 'b>(tanks_render_data: &'a TanksRenderData, primary_render_pass: &'b mut wgpu::RenderPass<'a>) {
    primary_render_pass.set_pipeline(&tanks_render_data.debug_lines_pipeline);

    for debug_lines in &tanks_render_data.debug_line_instances {
        primary_render_pass.set_bind_group(1, &debug_lines.color_bind_group, &[]);
        primary_render_pass.draw_mesh(&debug_lines.mesh, &tanks_render_data.render_camera.camera_bind_group);
    }
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