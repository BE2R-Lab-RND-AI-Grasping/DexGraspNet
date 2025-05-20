import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import json


class ContactPointPicker:
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self.picked_points = []

        # GUI и окно
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Выбор контактных точек", 1024, 768)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        # Загрузка меша
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.mesh.compute_vertex_normals()
        self.tmesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        self.scene.scene.add_geometry("mesh", self.mesh, self.material)
        bounds = self.mesh.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bounds, bounds.get_center())

        # Raycasting сцена
        self.raycasting_scene = o3d.t.geometry.RaycastingScene()
        self.mesh_id = self.raycasting_scene.add_triangles(self.tmesh)

        # Обработка кликов мыши
        self.scene.set_on_mouse(self.on_mouse_event)

    def on_mouse_event(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and (event.buttons & int(gui.MouseButton.LEFT)):
            x = event.x
            y = event.y

            try:
                start = np.asarray(self.scene.scene.camera.unproject(x, y, 0.1, self.scene.frame.width, self.scene.frame.height))
                end = np.asarray(self.scene.scene.camera.unproject(x, y, 0.9, self.scene.frame.width, self.scene.frame.height))
            except Exception as e:
                print("⚠️ Ошибка при проекции:", e)
                return gui.Widget.EventCallbackResult.IGNORED

            direction = end - start
            norm = np.linalg.norm(direction)
            if norm == 0:
                print("❌ Нулевое направление луча")
                return gui.Widget.EventCallbackResult.IGNORED
            direction /= norm

            rays = o3d.core.Tensor(np.array([[start[0], start[1], start[2], direction[0], direction[1], direction[2]]], dtype=np.float32))
            ans = self.raycasting_scene.cast_rays(rays)

            t_hit = ans['t_hit'].numpy()[0]
            if np.isfinite(t_hit):
                point = start + t_hit * direction
                self.picked_points.append(point.tolist())
                print("✅ Контактная точка:", point)

                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
                sphere.paint_uniform_color([1.0, 0.0, 0.0])
                sphere.translate(point)
                self.scene.scene.add_geometry(f"picked_{len(self.picked_points)}", sphere, self.material)

                return gui.Widget.EventCallbackResult.HANDLED

            print("❌ Не удалось найти пересечение.")
        return gui.Widget.EventCallbackResult.IGNORED

    def run(self):
        gui.Application.instance.run()

    def save_points(self, filename="contact_points.json"):
        rounded = [[round(coord, 4) for coord in point] for point in self.picked_points]
        with open(filename, "w") as f:
            json.dump(rounded, f, indent=2)
        print(f"💾 Контактные точки сохранены в {filename}")


# ==== Запуск ====
if __name__ == "__main__":
    picker = ContactPointPicker("assets/Link_thumb_finray_proxy.STL")  # замени путь на свой
    picker.run()
    picker.save_points()