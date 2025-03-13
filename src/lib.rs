use glam::{Mat4, Vec2, Vec3};
use wgpu::util::DeviceExt;
use xenofrost::core::{app::App, input_manager::InputManager, render_engine::{camera::{Camera, CameraBindGroupLayout, CameraProjection, OrthographicProjection}, mesh::QuadMesh, pipeline::Pipeline2D, AspectRatio, DrawMesh, InstanceRaw, PrimaryRenderPass, RenderEngine}, world::{component::Component, query_resource, resource::Resource, world_query, Transform2D, World}};

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
    app.add_prepare_system(Box::new(camera_prepare_system));
    app.add_prepare_system(Box::new(circle_prepare_system));
    app.add_render_system(Box::new(circles_render_system));

    app.run();
}

#[derive(Resource)]
pub struct RenderCircleInstances {
    pub instances: Vec<InstanceRaw>,
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

#[derive(Component)]
pub struct RenderCircle;

fn startup_system(world: &mut World) {
    let render_engine = query_resource!(world, RenderEngine).unwrap();

    let quad_mesh = QuadMesh::new(&render_engine.data().device);
    world.add_resource(quad_mesh);

    let camera_bind_group_layout = CameraBindGroupLayout::new(&render_engine);
    world.add_resource(camera_bind_group_layout);
    
    let pipeline2d = Pipeline2D::new(world);
    world.add_resource(pipeline2d);
    world.add_resource(RenderCircleInstances::new(&render_engine.data().device));

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
    world.add_component_to_entity(circle, RenderCircle);
    world.add_component_to_entity(circle, Transform2D {
        translation: Vec2::new(0.0, 0.0),
        scale: Vec2::new(1.0, 1.0),
        rotation: 0.0
    });
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
    for (_, tranform2d, _) in circles_query(world).iter() {
        let raw_instance = InstanceRaw {
            model: Mat4::from_translation(Vec3::new(tranform2d.translation.x, tranform2d.translation.y, 0.0)).to_cols_array_2d()
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
    let pipeline2d = query_resource!(world, Pipeline2D).unwrap();
    let circle_instances = query_resource!(world, RenderCircleInstances).unwrap();
    let quad_mesh_handle = query_resource!(world, QuadMesh).unwrap();
    let quad_mesh = quad_mesh_handle.data();
    let primary_render_pass = query_resource!(world, PrimaryRenderPass).unwrap();

    let camera_query = world_query!(Camera);
    let camera_query_invoke = camera_query(world);
    let (_, camera) = camera_query_invoke.iter().next().unwrap();

    primary_render_pass.data_mut().render_pass.as_mut().unwrap().set_pipeline(&pipeline2d.data().pipeline);
    primary_render_pass.data_mut().render_pass.as_mut().unwrap().set_vertex_buffer(1, circle_instances.data().instances_buffer.slice(..));
    primary_render_pass.data_mut().render_pass.as_mut().unwrap().draw_mesh_instanced(&quad_mesh.mesh, 0..1 as u32, &camera.camera_bind_group);
}