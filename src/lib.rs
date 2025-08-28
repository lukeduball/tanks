use std::vec;

use glam::{IVec2, Mat4, Vec2, Vec3};
use wgpu::{util::DeviceExt, BindGroupDescriptor, BindGroupEntry};
use xenofrost::{core::{app::App, input_manager::{InputManager, KeyCode}, math::bounding2d::Polygon2d, render_engine::{camera::{Camera, CameraProjection, OrthographicProjection}, mesh::{AtlasQuadMesh, Mesh}, pipeline::{AtlasPipeline2d, DebugLineInstance, DebugLinesPipeline2d, InstanceAtlas}, texture::{Texture, TextureBindGroupLayout}, AspectRatio, DrawMesh, PrimaryRenderPass, RenderEngine}, world::{component::Component, query_resource, resource::Resource, world_query, Collider2d, Colliders2d, Transform2d, World}}, include_bytes_from_project_path};

const BASELINE_NUMBER_OF_RESOURCES: u64 = xenofrost::NUMBER_OF_RESOURCES;
const BASELINE_NUMBER_OF_COMPONENTS: u64 = xenofrost::NUMBER_OF_COMPONENTS;

const HALF_WORLD_WIDTH: f32 = 5.0;
const HALF_WORLD_HEIGHT: f32 = 5.0;

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub fn run() {
    cfg_if::cfg_if!(
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Unable to initialize logger!");
        }
    );

    let mut app = App::new("tanks");
    app.add_startup_system(Box::new(startup_system));
    app.add_update_system(Box::new(world_border_system));
    app.add_update_system(Box::new(player_tank_controller_system));
    app.add_update_system(Box::new(update_tank_bounding_boxes));
    app.add_prepare_system(Box::new(camera_prepare_system));
    app.add_prepare_system(Box::new(tanks_prepare_system));
    app.add_prepare_system(Box::new(debug_shapes_prepare_system));
    app.add_render_system(Box::new(tanks_render_system));
    app.add_render_system(Box::new(debug_shapes_render_system));

    app.run();
}

#[derive(Resource)]
struct DebugLineInstances {
    instances: Vec<DebugLineInstance>,
}

impl DebugLineInstances {
    fn new() -> Self {
        let instances = Vec::new();

        Self {
            instances,
        }
    }
}

#[derive(Resource)]
pub struct RenderTankInstances {
    pub instances: Vec<InstanceAtlas>,
    pub prev_size: usize,
    pub instances_buffer: wgpu::Buffer,
}

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

impl RenderTankInstances {
    pub fn new(device: &wgpu::Device) -> Self {
        let instances = Vec::new();
        let instances_buffer = device.create_buffer(&wgpu::BufferDescriptor { 
            label: Some("Tank Instances"), 
            size: 1, 
            usage: wgpu::BufferUsages::VERTEX, 
            mapped_at_creation: false 
        });

        let prev_size = instances.len();

        Self {
            instances,
            prev_size,
            instances_buffer
        }
    }
}

#[derive(Resource)]
struct TanksTextureAtlasBindGroup {
    _texture: Texture,
    bind_group: wgpu::BindGroup
}

impl TanksTextureAtlasBindGroup {
    fn new(world: &mut World) -> Self {
        let render_engine = query_resource!(world, RenderEngine).unwrap();
        let texture_bind_group_layout = query_resource!(world, TextureBindGroupLayout).unwrap();

        let texture_atlas = Texture::from_bytes(
            &render_engine.data().device, 
            &render_engine.data().queue, 
            include_bytes_from_project_path!("/res/game_objects/tank_texture_atlas.png"), 
            "Tanks Texture Atlas"
        );

        Self {
            bind_group: render_engine.data().device.create_bind_group(&BindGroupDescriptor {
                label: Some("Tanks Texture Atlas Bind Group"),
                layout: &texture_bind_group_layout.data().bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_atlas.view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture_atlas.sampler),
                    }
                ],
            }),
            _texture: texture_atlas,
        }
    }
}

#[derive(Component)]
pub struct RenderTank {
    cannon_rotation: f32
}

#[derive(Component)]
struct PlayerController;

fn startup_system(world: &mut World) {
    let render_engine = query_resource!(world, RenderEngine).unwrap();

    let input_manager = query_resource!(world, InputManager).unwrap();
    input_manager.data_mut().register_key_binding("up", KeyCode::KeyW);
    input_manager.data_mut().register_key_binding("down", KeyCode::KeyS);
    input_manager.data_mut().register_key_binding("left", KeyCode::KeyA);
    input_manager.data_mut().register_key_binding("right", KeyCode::KeyD);
    input_manager.data_mut().register_key_binding("space", KeyCode::Space);

    let atlas_quad_mesh = AtlasQuadMesh::new(&render_engine.data().device);
    world.add_resource(atlas_quad_mesh);
    
    let debug_lines_pipeline2d = DebugLinesPipeline2d::new(world);
    world.add_resource(debug_lines_pipeline2d);

    let pipeline2d = AtlasPipeline2d::new(world);
    world.add_resource(pipeline2d);
    world.add_resource(RenderTankInstances::new(&render_engine.data().device));

    world.add_resource(DebugLineInstances::new());

    let tanks_texture_atlas_bind_group = TanksTextureAtlasBindGroup::new(world);
    world.add_resource(tanks_texture_atlas_bind_group);

    let aspect_ratio = query_resource!(world, AspectRatio).unwrap();

    let camera_entity = world.spawn_entity();
    world.add_component_to_entity(camera_entity, Transform2d {
        translation: Vec2::new(0.0, 0.0),
        scale: Vec2::new(1.0, 1.0),
        rotation: 0.0
    });
    let camera_component = Camera::new(
        "primary_camera", 
        CameraProjection::Orthographic(OrthographicProjection {
            width: 10.0,
            height: 10.0,
            near_clip: 0.1,
            far_clip: 1000.0,
            aspect_ratio: aspect_ratio.data().aspect_ratio
        }), 
        world
    );
    world.add_component_to_entity(camera_entity, camera_component);

    let player_tank = world.spawn_entity();
    world.add_component_to_entity(player_tank, RenderTank {cannon_rotation: 0.0});
    world.add_component_to_entity(player_tank, Transform2d {
        translation: Vec2::new(0.0, 0.0),
        scale: Vec2::new(1.0, 1.0),
        rotation: 0.0
    });
    world.add_component_to_entity(player_tank, PlayerController);
    let mut colliders = Colliders2d::new();
    colliders.collider_list.push(Collider2d::new(
        Polygon2d::new(TANK_COLLISION_POINTS[0].to_vec()), 
        Transform2d { 
            translation: Vec2::splat(0.0), 
            scale: Vec2::splat(1.0), 
            rotation: 0.0 
        })
    );
    world.add_component_to_entity(player_tank, colliders);
}

fn player_tank_controller_system(world: &mut World) {
    let rotation_speed = 0.5;
    let movement_speed = 0.005;

    let render_engine = query_resource!(world, RenderEngine).unwrap();
    let input_manager_handle = query_resource!(world, InputManager).unwrap();
    let player_tank_query = world_query!(mut Transform2d, PlayerController, mut RenderTank);
    let player_tank_query_invoke = player_tank_query(world);
    let (_, mut transform2d, _, mut cannon) = player_tank_query_invoke.iter().next().unwrap();

    let camera_query = world_query!(Transform2d, Camera);
    let camera_query_invoke = camera_query(world);
    let (_, camera_transform2d, camera) = camera_query_invoke.iter().next().unwrap();

    let input_manager = input_manager_handle.data();
    let left_key_state = input_manager.get_key_state("left").unwrap();
    let right_key_state = input_manager.get_key_state("right").unwrap();
    let up_key_state = input_manager.get_key_state("up").unwrap();
    let down_key_state = input_manager.get_key_state("down").unwrap();
    let space_key_state = input_manager.get_key_state("space").unwrap();

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

        let current_rotation = transform2d.rotation;

        let mut rotation_diff_degrees = target_degrees_constrained - current_rotation;
        let rotation_diff_degrees_abs = f32::abs(rotation_diff_degrees);
        if rotation_diff_degrees_abs > 180.0 {rotation_diff_degrees *= -1.0; }

        if rotation_diff_degrees_abs < 0.0001 {
            transform2d.set_rotation(target_degrees_constrained);
        }
        else {
            transform2d.rotate(rotation_diff_degrees.signum() * rotation_speed);
        }

        let movement_vector = Vec2::new(transform2d.rotation.to_radians().cos(), transform2d.rotation.to_radians().sin());
        transform2d.translation += movement_vector * movement_speed; 
    }

    let window_size = IVec2::new(render_engine.data().window_width as i32, render_engine.data().window_height as i32);
    let pixel_mouse_coords = input_manager.get_mouse_physical();
    let mouse_coords = camera.convert_screen_space_to_camera_space(Vec3::new(camera_transform2d.translation.x, camera_transform2d.translation.y, 0.0), Vec3::new(0.0, 0.0, 1.0), pixel_mouse_coords, window_size);
    let cannon_center = transform2d.translation + Vec2::new(0.0, 0.01875);
    let tank_mouse_difference = Vec2::new(mouse_coords.x, mouse_coords.y) - cannon_center;
    let rotation = (f32::atan2(tank_mouse_difference.y, tank_mouse_difference.x).to_degrees() + 360.0) % 360.0;
    cannon.cannon_rotation = rotation;
}

fn update_tank_bounding_boxes(world: &mut World) {
    let tank_bounding_box_query = world_query!(Transform2d, RenderTank, mut Colliders2d);
    for (_, transform, _canon, mut obb_list) in tank_bounding_box_query(world).iter() {
        let atlas_index = get_tank_atlas_index(transform.rotation);
        let collider_translation = obb_list.collider_list[0].transform.translation;
        obb_list.collider_list[0].polygon2d = Polygon2d::new(TANK_COLLISION_POINTS[((atlas_index+4)%8) as usize].to_vec());
        obb_list.collider_list[0].polygon2d.set_translation_rotation(transform.translation + collider_translation, 0.0);
    }
}

fn world_border_system(world: &mut World) {
    let tanks_query = world_query!(mut Transform2d, RenderTank);
    for (_, mut transform2d, _) in tanks_query(&world).iter() {
        if transform2d.translation.x <= -HALF_WORLD_WIDTH {
            transform2d.translation.x = -HALF_WORLD_WIDTH;
        }

        if transform2d.translation.x >= HALF_WORLD_WIDTH {
            transform2d.translation.x = HALF_WORLD_WIDTH;
        }

        if transform2d.translation.y <= -HALF_WORLD_HEIGHT {
            transform2d.translation.y = -HALF_WORLD_HEIGHT;
        }

        if transform2d.translation.y >= HALF_WORLD_HEIGHT {
            transform2d.translation.y = HALF_WORLD_HEIGHT;
        }
    }
}

fn camera_prepare_system(world: &mut World) {
    let render_engine = query_resource!(world, RenderEngine).unwrap();

    let camera_query = world_query!(Transform2d, mut Camera);
    if let Some((_, transform2d, mut camera)) = camera_query(world).iter().next() {
        camera.update_uniform_buffer(
            Vec3::new(transform2d.translation.x, transform2d.translation.y, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
            &render_engine.data().queue
        );
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

fn tanks_prepare_system(world: &mut World) {
    let render_engine = query_resource!(world, RenderEngine).unwrap();
    let tank_instances = query_resource!(world, RenderTankInstances).unwrap();
    let tanks_query = world_query!(Transform2d, RenderTank);

    tank_instances.data_mut().instances.clear();
    for (_, transform2d, render_tank) in tanks_query(world).iter() {
        let base_atlas_index = get_tank_atlas_index(transform2d.rotation);
        let base_atlas_tex_coords_x = base_atlas_index % 16;
        let base_atlas_tex_coords_y = base_atlas_index / 16;
        let base_raw_instance = InstanceAtlas {
            model: Mat4::from_translation(Vec3::new(transform2d.translation.x, transform2d.translation.y, 0.0)),
            tex_coords: Vec2::new(0.0625*base_atlas_tex_coords_x as f32, 0.0625*base_atlas_tex_coords_y as f32),
            sprite_size: Vec2::new(0.0625, 0.0625)
        };
        //Add tank base instance
        tank_instances.data_mut().instances.push(base_raw_instance);

        let cannon_atlas_index = get_tank_cannon_atlas_index(render_tank.cannon_rotation);
        let cannon_atlas_tex_coords_x = cannon_atlas_index % 16;
        let cannon_atlas_tex_coords_y = cannon_atlas_index / 16;
        let cannon_raw_instance = InstanceAtlas {
            model: Mat4::from_translation(Vec3::new(transform2d.translation.x, transform2d.translation.y, 0.1)),
            tex_coords: Vec2::new(0.0625*cannon_atlas_tex_coords_x as f32, 0.0625*cannon_atlas_tex_coords_y as f32),
            sprite_size: Vec2::new(0.0625, 0.0625)
        };
        //Add tank cannon instance
        tank_instances.data_mut().instances.push(cannon_raw_instance);
    }

    if tank_instances.data().instances.len() != tank_instances.data().prev_size {
        tank_instances.data_mut().instances_buffer.destroy();
        let new_instances_buffer = render_engine.data().device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tanks Instance Buffer"),
            contents: bytemuck::cast_slice(&tank_instances.data().instances),
            usage: wgpu::BufferUsages::VERTEX
        });
        tank_instances.data_mut().instances_buffer = new_instances_buffer;
    }
    else {
        render_engine.data().queue.write_buffer(&tank_instances.data().instances_buffer, 0, bytemuck::cast_slice(&tank_instances.data().instances));
    }
}

fn tanks_render_system(world: &mut World) {
    let pipeline2d = query_resource!(world, AtlasPipeline2d).unwrap();
    let tank_instances = query_resource!(world, RenderTankInstances).unwrap();
    let atlas_quad_mesh_handle = query_resource!(world, AtlasQuadMesh).unwrap();
    let atlas_quad_mesh = atlas_quad_mesh_handle.data();
    let primary_render_pass = query_resource!(world, PrimaryRenderPass).unwrap();

    let texture_atlas_bind_group = query_resource!(world, TanksTextureAtlasBindGroup).unwrap();

    let camera_query = world_query!(Camera);
    let camera_query_invoke = camera_query(world);
    let (_, camera) = camera_query_invoke.iter().next().unwrap();

    primary_render_pass.data_mut().render_pass.as_mut().unwrap().set_pipeline(&pipeline2d.data().pipeline);
    primary_render_pass.data_mut().render_pass.as_mut().unwrap().set_vertex_buffer(1, tank_instances.data().instances_buffer.slice(..));
    primary_render_pass.data_mut().render_pass.as_mut().unwrap().set_bind_group(1, &texture_atlas_bind_group.data().bind_group, &[]);
    primary_render_pass.data_mut().render_pass.as_mut().unwrap().draw_mesh_instanced(&atlas_quad_mesh.mesh, 0..tank_instances.data().instances.len() as u32, &camera.camera_bind_group);
}

fn debug_shapes_prepare_system(world: &mut World) {
    let render_engine = query_resource!(world, RenderEngine).unwrap();
    let colliders_query = world_query!(Colliders2d);
    let debug_line_instances = query_resource!(world, DebugLineInstances).unwrap();

    // The existing vertex and instance buffers will be cleaned up because wgpu::Buffer implements the Drop trait
    debug_line_instances.data_mut().instances.clear();
    for (entity, colliders) in colliders_query(world).iter() {
        for collider2d in &colliders.collider_list {

            let vertices: Vec<Vec3> = collider2d.polygon2d.points.clone().into_iter().map(|vec2| Vec3::new(vec2.x, vec2.y, 10.0)).collect();
            let vertex_buffer = render_engine.data().device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(format!("Vertex Buffer for {}", entity).as_str()),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let mut index_list: Vec<u16> = (0..collider2d.polygon2d.points.len() as u16).collect();
            index_list.push(0);
            let index_buffer = render_engine.data().device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(format!("Index Buffer for {}", entity).as_str()),
                contents: bytemuck::cast_slice(&index_list),
                usage: wgpu::BufferUsages::INDEX,
            });

            let num_indices = collider2d.polygon2d.points.len() + 1;
            
            let mesh = Mesh {
                name: format!("Debug Line for {}", entity),
                vertex_buffer: vertex_buffer,
                index_buffer: index_buffer,
                num_elements: num_indices as u32,
            };
            debug_line_instances.data_mut().instances.push(DebugLineInstance { 
                mesh 
            });
        }
    }
}

fn debug_shapes_render_system(world: &mut World) {
    let debug_border_instances = query_resource!(world, DebugLineInstances).unwrap();
    let debug_lines_pipeline2d = query_resource!(world, DebugLinesPipeline2d).unwrap();
    let primary_render_pass = query_resource!(world, PrimaryRenderPass).unwrap();

    let camera_query = world_query!(Camera);
    let camera_query_invoke = camera_query(world);
    let (_, camera) = camera_query_invoke.iter().next().unwrap();

    primary_render_pass.data_mut().render_pass.as_mut().unwrap().set_pipeline(&debug_lines_pipeline2d.data().pipeline);

    for debug_line_instance in &debug_border_instances.data().instances {
        primary_render_pass.data_mut().render_pass.as_mut().unwrap().draw_mesh(&debug_line_instance.mesh, &camera.camera_bind_group);
    }
}