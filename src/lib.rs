use std::f32::consts::PI;

use glam::{IVec2, Mat4, Vec2, Vec3};
use wgpu::{util::DeviceExt, BindGroupDescriptor, BindGroupEntry};
use xenofrost::{core::{app::App, input_manager::{InputManager, KeyCode}, render_engine::{camera::{Camera, CameraProjection, OrthographicProjection}, mesh::AtlasQuadMesh, pipeline::{AtlasPipeline2D, InstanceAtlas}, texture::{Texture, TextureBindGroupLayout}, AspectRatio, DrawMesh, PrimaryRenderPass, RenderEngine}, world::{component::Component, query_resource, resource::Resource, world_query, Transform2D, World}}, include_bytes_from_project_path};

const BASELINE_NUMBER_OF_RESOURCES: u64 = xenofrost::NUMBER_OF_RESOURCES;
const BASELINE_NUMBER_OF_COMPONENTS: u64 = xenofrost::NUMBER_OF_COMPONENTS;

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
    app.add_update_system(Box::new(circles_atlas_update_system));
    app.add_update_system(Box::new(player_tank_controller_system));
    app.add_prepare_system(Box::new(camera_prepare_system));
    app.add_prepare_system(Box::new(tanks_prepare_system));
    app.add_render_system(Box::new(tanks_render_system));

    app.run();
}

#[derive(Resource)]
pub struct RenderTankInstances {
    pub instances: Vec<InstanceAtlas>,
    pub prev_size: usize,
    pub instances_buffer: wgpu::Buffer,
}

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
    atlas_index: u32
}

#[derive(Component)]
struct PlayerController;

impl RenderTank {
    fn new(atlas_index: u32) -> Self {
        Self {atlas_index}
    }
}

fn startup_system(world: &mut World) {
    let render_engine = query_resource!(world, RenderEngine).unwrap();

    let input_manager = query_resource!(world, InputManager).unwrap();
    input_manager.data_mut().register_key_binding("up", KeyCode::KeyW);
    input_manager.data_mut().register_key_binding("down", KeyCode::KeyS);
    input_manager.data_mut().register_key_binding("left", KeyCode::KeyA);
    input_manager.data_mut().register_key_binding("right", KeyCode::KeyD);
    input_manager.data_mut().register_key_binding("atlas_toggle", KeyCode::Space);

    let quad_mesh = AtlasQuadMesh::new(&render_engine.data().device);
    world.add_resource(quad_mesh);
    
    let pipeline2d = AtlasPipeline2D::new(world);
    world.add_resource(pipeline2d);
    world.add_resource(RenderTankInstances::new(&render_engine.data().device));

    let tanks_texture_atlas_bind_group = TanksTextureAtlasBindGroup::new(world);
    world.add_resource(tanks_texture_atlas_bind_group);

    let aspect_ratio = query_resource!(world, AspectRatio).unwrap();

    let camera_entity = world.spawn_entity();
    world.add_component_to_entity(camera_entity, Transform2D {
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
    world.add_component_to_entity(player_tank, RenderTank::new(0));
    world.add_component_to_entity(player_tank, Transform2D {
        translation: Vec2::new(0.0, 0.0),
        scale: Vec2::new(1.0, 1.0),
        rotation: 0.0
    });
    world.add_component_to_entity(player_tank, PlayerController);
}

fn circles_atlas_update_system(world: &mut World) {
    let input_manager_handle = query_resource!(world, InputManager).unwrap();
    let input_manager = input_manager_handle.data();
    let atlas_toggle_key_state = input_manager.get_key_state("atlas_toggle").unwrap();

    let atlas_object_query = world_query!(mut RenderTank);
    for (_, mut atlas_object) in atlas_object_query(world).iter() {
        if atlas_toggle_key_state.get_was_pressed() {
            atlas_object.atlas_index += 1;
        }
    }
}

fn player_tank_controller_system(world: &mut World) {
    let rotation_speed = 0.1;
    let movement_speed = 0.001;

    let input_manager_handle = query_resource!(world, InputManager).unwrap();
    let player_tank_query = world_query!(mut Transform2D, PlayerController, RenderTank);
    let player_tank_query_invoke = player_tank_query(world);
    let (_, mut transform2d, _, _) = player_tank_query_invoke.iter().next().unwrap();

    let input_manager = input_manager_handle.data();
    let left_key_state = input_manager.get_key_state("left").unwrap();
    let right_key_state = input_manager.get_key_state("right").unwrap();
    let up_key_state = input_manager.get_key_state("up").unwrap();
    let down_key_state = input_manager.get_key_state("down").unwrap();
    let print_rotation_key = input_manager.get_key_state("atlas_toggle").unwrap();

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

    if print_rotation_key.get_was_pressed() {
        println!("{}", transform2d.rotation);
    }
}

fn camera_prepare_system(world: &mut World) {
    let render_engine = query_resource!(world, RenderEngine).unwrap();

    let camera_query = world_query!(Transform2D, mut Camera);
    if let Some((_, transform2d, mut camera)) = camera_query(world).iter().next() {
        camera.update_uniform_buffer(
            Vec3::new(transform2d.translation.x, transform2d.translation.y, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
            &render_engine.data().queue
        );
    }
}

fn tanks_prepare_system(world: &mut World) {
    let render_engine = query_resource!(world, RenderEngine).unwrap();
    let tank_instances = query_resource!(world, RenderTankInstances).unwrap();
    let tanks_query = world_query!(Transform2D, RenderTank);

    tank_instances.data_mut().instances.clear();
    for (_, tranform2d, atlas_object) in tanks_query(world).iter() {
        let atlas_tex_coords_x = atlas_object.atlas_index % 16;
        let atlas_tex_coords_y = atlas_object.atlas_index / 16;
        let raw_instance = InstanceAtlas {
            model: Mat4::from_translation(Vec3::new(tranform2d.translation.x, tranform2d.translation.y, 0.0)),
            tex_coords: Vec2::new(0.0625*atlas_tex_coords_x as f32, 0.0625*atlas_tex_coords_y as f32),
            sprite_size: Vec2::new(0.0625, 0.0625)
        };
        tank_instances.data_mut().instances.push(raw_instance);
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
    let pipeline2d = query_resource!(world, AtlasPipeline2D).unwrap();
    let tank_instances = query_resource!(world, RenderTankInstances).unwrap();
    let quad_mesh_handle = query_resource!(world, AtlasQuadMesh).unwrap();
    let quad_mesh = quad_mesh_handle.data();
    let primary_render_pass = query_resource!(world, PrimaryRenderPass).unwrap();

    let texture_atlas_bind_group = query_resource!(world, TanksTextureAtlasBindGroup).unwrap();

    let camera_query = world_query!(Camera);
    let camera_query_invoke = camera_query(world);
    let (_, camera) = camera_query_invoke.iter().next().unwrap();

    primary_render_pass.data_mut().render_pass.as_mut().unwrap().set_pipeline(&pipeline2d.data().pipeline);
    primary_render_pass.data_mut().render_pass.as_mut().unwrap().set_vertex_buffer(1, tank_instances.data().instances_buffer.slice(..));
    primary_render_pass.data_mut().render_pass.as_mut().unwrap().set_bind_group(1, &texture_atlas_bind_group.data().bind_group, &[]);
    primary_render_pass.data_mut().render_pass.as_mut().unwrap().draw_mesh_instanced(&quad_mesh.mesh, 0..1 as u32, &camera.camera_bind_group);
}