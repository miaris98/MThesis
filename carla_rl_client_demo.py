import os
import sys
import time
import math
import argparse
import random
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="CARLA RL Client Demo - Control Vehicle, Save Screenshots & Calculate Rewards.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA server IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port (default: 2000)")
    parser.add_argument("--steps", type=int, default=50, help="Number of simulation steps to execute")
    parser.add_argument("--output-dir", type=str, default="output_screenshots", help="Directory to save screenshots")
    parser.add_argument("--img-width", type=int, default=640, help="Camera width")
    parser.add_argument("--img-height", type=int, default=480, help="Camera height")
    
    args = parser.parse_args()

    try:
        import carla
    except ImportError:
        print("Error: 'carla' python package is not installed. Please run 'pip install carla'.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Screenshots will be saved to: {os.path.abspath(args.output_dir)}")

    actor_list = []
    latest_image_data = {"frame": None, "array": None}
    collision_event = {"collided": False}

    try:
        print(f"Connecting to CARLA server at {args.host}:{args.port}...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Configure synchronous mode for reproducible RL stepping
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # 1. Spawn Vehicle
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print(f"Spawned Vehicle '{vehicle.type_id}' (ID: {vehicle.id}) at {spawn_point.location}")

        # 2. Attach RGB Camera Sensor
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(args.img_width))
        camera_bp.set_attribute("image_size_y", str(args.img_height))
        camera_bp.set_attribute("fov", "90")
        
        # Position camera slightly above and in front of the vehicle center
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        def camera_callback(image):
            # Convert raw CARLA image buffer to uint8 RGB array
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3] # Drop alpha channel (BGRA -> BGR)
            latest_image_data["frame"] = image.frame
            latest_image_data["array"] = array

        camera.listen(camera_callback)

        # 3. Attach Collision Sensor
        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        actor_list.append(collision_sensor)

        def collision_callback(event):
            collision_event["collided"] = True
            print(f"💥 COLLISION DETECTED with {event.other_actor.type_id}")

        collision_sensor.listen(collision_callback)

        # Allow initial tick for sensors to initialize
        world.tick()
        time.sleep(0.5)

        print("\n--- Starting RL Control Loop ---")
        total_reward = 0.0

        for step in range(1, args.steps + 1):
            # Generate continuous action (e.g. constant throttle, slight steering curve)
            throttle = 0.6
            steer = math.sin(step * 0.1) * 0.2  # Dynamic steering curve
            brake = 0.0

            # Apply action to vehicle
            control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            vehicle.apply_control(control)

            # Advance simulation world step
            world.tick()

            # Calculate vehicle velocity & speed (km/h)
            velocity = vehicle.get_velocity()
            speed_m_s = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_kmh = 3.6 * speed_m_s

            # Calculate RL Reward
            # Base reward: speed incentive minus penalties for sharp steering and collisions
            collision_penalty = 100.0 if collision_event["collided"] else 0.0
            steering_penalty = abs(steer) * 2.0
            
            step_reward = (speed_kmh * 0.1) - steering_penalty - collision_penalty
            total_reward += step_reward

            # Save camera screenshot if frame captured
            img_path = ""
            if latest_image_data["array"] is not None:
                # Save frame using PIL/OpenCV or raw writing
                try:
                    from PIL import Image
                    img = Image.fromarray(latest_image_data["array"][:, :, ::-1]) # Convert BGR to RGB
                    img_filename = f"step_{step:03d}.png"
                    img_path = os.path.join(args.output_dir, img_filename)
                    img.save(img_path)
                except ImportError:
                    pass

            print(
                f"[Step {step:02d}/{args.steps}] "
                f"Action(T={throttle:.1f}, S={steer:+.2f}) | "
                f"Speed: {speed_kmh:5.1f} km/h | "
                f"Reward: {step_reward:+.2f} | "
                f"Total Reward: {total_reward:+.2f} | "
                f"Collision: {collision_event['collided']} | "
                f"Frame Saved: {os.path.basename(img_path) if img_path else 'Pending'}"
            )

            # Reset collision status for next step
            collision_event["collided"] = False
            time.sleep(0.02)

        print("\n--- Simulation Episode Finished ---")
        print(f"Total Cumulative Reward: {total_reward:.2f}")

    except Exception as e:
        print(f"\nExecution error: {e}")
    finally:
        # Restore original world settings
        try:
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except Exception:
            pass

        print("Cleaning up spawned actors...")
        for actor in reversed(actor_list):
            try:
                actor.destroy()
            except Exception:
                pass
        print("Cleanup completed.")

if __name__ == "__main__":
    main()
