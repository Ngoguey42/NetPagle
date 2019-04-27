import numpy as np

from constants import Offset, MAGIC_SCALE_FACTOR

class Camera():
    """Only work with 1920x1080 and windowed-borderless options"""
    def __init__(self, w):
        cam0 = w.pm.read_uint(w.base_address + Offset.camera)
        cam1 = w.pm.read_uint(cam0 + Offset.Camera.offset)

        self.xyz = w.pull_floats(cam1 + Offset.Camera.xyz, (3,))
        self.facing = w.pull_floats(cam1 + Offset.Camera.facing, (3, 3))
        self.fov = w.pull_floats(cam1 + Offset.Camera.fov, ())
        self.aspect = w.pull_floats(cam1 + Offset.Camera.aspect, ())
        self.size = self.get_screen_size(w)

        aspect_error = abs(self.aspect - np.divide.reduce(self.size.astype(float))) / self.aspect
        assert aspect_error < 0.005, aspect_error

    @staticmethod
    def get_screen_size(w):
        addresses = [
            # Screen size was found at all those addresses.
            # If one of those addresses make the script crash, comment it.
            0x00884E28,
            0x00CE8B60,
            0x00CE8B68,
            0x00CE8B7C,
            0x00CE8B84,
            0x00CE8B9C,
            0x00CE8BA4,
        ]
        sizes = {
            a: w.pull_u32s(a, (2,))
            for a in addresses
        }
        for a, s in sizes.items():
            assert np.all(sizes[addresses[0]] == s), (
                '\n'
                f'{a:#x} says {s}'
                '\n'
                f'{addresses[0]:#x} says {sizes[addresses[0]]}'
            )
        return sizes[addresses[0]]

    def world_to_screen(self, xyz):
        diff = xyz - self.xyz
        """
        At this point:
        - Origin is the center of the screen
        - Unit vector is a yard long
        - Axes are right handed
               z-axis (sky)
                  ^
                  |  7 x-axis (north)
                  | /
         y-axis   |/
          <-------+
        (west)
        """

        view = diff @ np.linalg.inv(self.facing)
        """
        At this point:
        - Origin is the center of the screen
        - Unit vector is ~a yard long
        - Axes are right handed
               z-axis (top of the screen)
                  ^
                  |  7 x-axis (depth)
                  | /
         y-axis   |/
          <-------+
        (left of the screen)
        """

        cam = np.asarray([-view[1], -view[2], view[0]])
        """
        At this point:
        - Origin is the center of the screen
        - Unit vector is a yard long
        - Axes are right handed
            7 z-axis (depth)
           /
          /      x-axis
         +--------->
         |    (right of the screen)
         |
         |
         v  y-axis
        (bottom of the screen)
        """

        fov_x = (1 / (1 + 1 / self.aspect ** 2)) ** 0.5
        fov_y = fov_x / self.aspect
        fov_x *= self.fov
        fov_y *= self.fov
        fov_x *= MAGIC_SCALE_FACTOR

        screen_right_at_unit_depth = np.tan(fov_x / 2)
        screen_bottom_at_unit_depth = np.tan(fov_y / 2)

        screen_right_at_point_depth = screen_right_at_unit_depth * cam[2]
        screen_bottom_at_point_depth = screen_bottom_at_unit_depth * cam[2]

        screen = np.asarray([
            cam[0] / screen_right_at_point_depth,
            cam[1] / screen_bottom_at_point_depth,
        ])
        """
        At this point:
        - Origin is the center of the screen
        - Unit vector is half of the screen
                 x-axis
         +--------->
         |    (right of the screen)
         |
         |
         v  y-axis
        (bottom of the screen)
        """

        raster = self.size / 2 * (1 + screen)
        """
        At this point:
        - Origin is the top left of the screen
        - Unit vector is a pixel
                 x-axis
         +--------->
         |    (right of the screen)
         |
         |
         v  y-axis
        (bottom of the screen)
        """

        behind = cam[2] < 0
        visible = np.all(np.abs(screen) <= 1) and not behind
        return raster, visible, behind
