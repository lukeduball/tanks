use glam::{Mat4, Vec2, Vec3};
use wgpu::{util::DeviceExt, BindGroupDescriptor, BindGroupEntry};
use xenofrost::{core::{app::App, input_manager::{InputManager}, render_engine::{camera::{Camera, CameraProjection, OrthographicProjection}, mesh::AtlasQuadMesh, pipeline::{AtlasPipeline2D, InstanceAtlas}, texture::{Texture, TextureBindGroupLayout}, AspectRatio, DrawMesh, PrimaryRenderPass, RenderEngine}, world::{component::Component, query_resource, resource::Resource, world_query, Transform2D, World}}, include_bytes_from_project_path};

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
    app.add_update_system(Box::new(camera_controller_system));
    app.add_update_system(Box::new(circles_atlas_update_system));
    app.add_prepare_system(Box::new(camera_prepare_system));
    app.add_prepare_system(Box::new(circle_prepare_system));
    app.add_render_system(Box::new(circles_render_system));

    app.run();
}

#[derive(Resource)]
pub struct RenderCircleInstances {
    pub instances: Vec<InstanceAtlas>,
    pub prev_size: usize,
    pub instances_buffer: wgpu::Buffer,
}

impl RenderCircleInstances {
    pub fn new(device: &wgpu::Device) -> Self {
        let instances = Vec::new();
        let instances_buffer = device.create_buffer(&wgpu::BufferDescriptor { 
            label: Some("Circle Instances"), 
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
pub struct RenderCircle {
    atlas_index: u32
}

impl RenderCircle {
    fn new(atlas_index: u32) -> Self {
        Self {atlas_index}
    }
}

fn startup_system(world: &mut World) {
    let render_engine = query_resource!(world, RenderEngine).unwrap();

    let quad_mesh = AtlasQuadMesh::new(&render_engine.data().device);
    world.add_resource(quad_mesh);
    
    let pipeline2d = AtlasPipeline2D::new(world);
    world.add_resource(pipeline2d);
    world.add_resource(RenderCircleInstances::new(&render_engine.data().device));

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

    let circle = world.spawn_entity();
    world.add_component_to_entity(circle, RenderCircle::new(0));
    world.add_component_to_entity(circle, Transform2D {
        translation: Vec2::new(0.0, 0.0),
        scale: Vec2::new(1.0, 1.0),
        rotation: 0.0
    });
}

fn circles_atlas_update_system(world: &mut World) {
    let input_manager_handle = query_resource!(world, InputManager).unwrap();
    let input_manager = input_manager_handle.data();
    let atlas_toggle_key_state = input_manager.get_key_state("atlas_toggle").unwrap();

    let atlas_object_query = world_query!(mut RenderCircle);
    for (_, mut atlas_object) in atlas_object_query(world).iter() {
        if atlas_toggle_key_state.get_was_pressed() {
            atlas_object.atlas_index += 1;
            println!("{}", atlas_object.atlas_index);
        }
    }
}

fn camera_controller_system(world: &mut World) {
    let speed = 0.01;

    let input_manager_handle = query_resource!(world, InputManager).unwrap();
    let camera_query = world_query!(mut Transform2D, Camera);
    let camera_query_invoke = camera_query(world);
    let (_, mut transform2d, _) = camera_query_invoke.iter().next().unwrap();

    let input_manager = input_manager_handle.data();
    let left_key_state = input_manager.get_key_state("left").unwrap();
    let right_key_state = input_manager.get_key_state("right").unwrap();
    let up_key_state = input_manager.get_key_state("up").unwrap();
    let down_key_state = input_manager.get_key_state("down").unwrap();

    if left_key_state.get_is_down() {
        transform2d.translation.x -= speed;
    }
    if right_key_state.get_is_down() {
        transform2d.translation.x += speed;
    }
    if up_key_state.get_is_down() {
        transform2d.translation.y += speed;
    }
    if down_key_state.get_is_down() {
        transform2d.translation.y -= speed;
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

fn circle_prepare_system(world: &mut World) {
    let render_engine = query_resource!(world, RenderEngine).unwrap();
    let circle_instances = query_resource!(world, RenderCircleInstances).unwrap();
    let circles_query = world_query!(Transform2D, RenderCircle);

    circle_instances.data_mut().instances.clear();
    for (_, tranform2d, atlas_object) in circles_query(world).iter() {
        let atlas_tex_coords_x = atlas_object.atlas_index % 16;
        let atlas_tex_coords_y = atlas_object.atlas_index / 16;
        let raw_instance = InstanceAtlas {
            model: Mat4::from_translation(Vec3::new(tranform2d.translation.x, tranform2d.translation.y, 0.0)),
            tex_coords: Vec2::new(0.0625*atlas_tex_coords_x as f32, 0.0625*atlas_tex_coords_y as f32),
            sprite_size: Vec2::new(0.0625, 0.0625)
        };
        circle_instances.data_mut().instances.push(raw_instance);
    }

    if circle_instances.data().instances.len() != circle_instances.data().prev_size {
        circle_instances.data_mut().instances_buffer.destroy();
        let new_instances_buffer = render_engine.data().device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Circle Instance Buffer"),
            contents: bytemuck::cast_slice(&circle_instances.data().instances),
            usage: wgpu::BufferUsages::VERTEX
        });
        circle_instances.data_mut().instances_buffer = new_instances_buffer;
    }
    else {
        render_engine.data().queue.write_buffer(&circle_instances.data().instances_buffer, 0, bytemuck::cast_slice(&circle_instances.data().instances));
    }


}

fn circles_render_system(world: &mut World) {
    let pipeline2d = query_resource!(world, AtlasPipeline2D).unwrap();
    let circle_instances = query_resource!(world, RenderCircleInstances).unwrap();
    let quad_mesh_handle = query_resource!(world, AtlasQuadMesh).unwrap();
    let quad_mesh = quad_mesh_handle.data();
    let primary_render_pass = query_resource!(world, PrimaryRenderPass).unwrap();

    let texture_atlas_bind_group = query_resource!(world, TanksTextureAtlasBindGroup).unwrap();

    let camera_query = world_query!(Camera);
    let camera_query_invoke = camera_query(world);
    let (_, camera) = camera_query_invoke.iter().next().unwrap();

    primary_render_pass.data_mut().render_pass.as_mut().unwrap().set_pipeline(&pipeline2d.data().pipeline);
    primary_render_pass.data_mut().render_pass.as_mut().unwrap().set_vertex_buffer(1, circle_instances.data().instances_buffer.slice(..));
    primary_render_pass.data_mut().render_pass.as_mut().unwrap().set_bind_group(1, &texture_atlas_bind_group.data().bind_group, &[]);
    primary_render_pass.data_mut().render_pass.as_mut().unwrap().draw_mesh_instanced(&quad_mesh.mesh, 0..1 as u32, &camera.camera_bind_group);
}