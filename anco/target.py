import numpy as np
import taichi as ti
from .utils import *

class Target():
    def __init__(self, chemical_id):
        self.chemical_id = chemical_id

    def init_taichi_objects(self):
        pass

    def get_scatter_function(self):
        return None

    def get_in_target_function(self, dt: float):
        return None
    
    def get_target_arrived_function(self):
        return None

class PointTarget(Target):
    def __init__(self, chemical_id, location, radius):
        self.location = location
        self.radius = radius
        self.chemical_id = chemical_id
    
    def init_taichi_objects(self):
        self.location_vector = ti.Vector(self.location)
        self.radius_field = Utils.create_single_float_field(self.radius)
    
    def get_scatter_function(self):
        @ti.kernel
        def scatter_in_target(particle_location: ti.template()):  # type: ignore
            for i in particle_location:
                f1 = ti.random()
                f2 = ti.random()
                if(f2 > f1):
                    f1, f2 = f2, f1
                f2 = f2 / f1 * 2. * ti.math.pi
                f1 *= self.radius_field[None]
                particle_location[i] = self.location_vector + ti.Vector([ti.cos(f2), ti.sin(f2)]) * f1
        return scatter_in_target

    def get_in_target_function(self, dt: float):
        @ti.func
        def in_target(x, y, do_not_consider_dt):
            return (x - self.location_vector[0]) ** 2 + (y - self.location_vector[1]) ** 2 < self.radius_field[None] ** 2
        return in_target
    
    def get_target_arrived_function(self):
        @ti.func
        def target_arrived(x, y):
            return 1.
        return target_arrived

class ImageTarget(Target):
    def __init__(self, chemical_id, image, remove_after_arrived=True, continuous_value=False, non_negative=True, diffusion=0., randomly_hit=False):
        self.chemical_id = chemical_id
        self.image = np.array(image)
        self.continuous_value = continuous_value
        self.diffusion = diffusion
        self.randomly_hit = randomly_hit
        if not continuous_value:
            mask = self.image > 0.5
            self.image[mask] = 1.
            self.image[np.logical_not(mask)] = 0.
        self.remove_after_arrived = remove_after_arrived
        self.non_negative = non_negative
    
    def init_taichi_objects(self):
        self.image_field = ti.field(dtype=ti.f32, shape=self.image.shape)
        self.image_field.from_numpy(np.array(self.image, dtype=np.float32))
    
    def get_scatter_function(self):
        number_of_pixels = np.sum(self.image)
        if number_of_pixels == 0:
            raise ValueError("Number of pixels of the ImageTarget object is 0.")
        number_of_pixels_field = Utils.create_single_float_field(number_of_pixels)
        threshold = Utils.create_single_float_field(np.min(self.image[self.image > 0.]) * 0.5)
        @ti.kernel
        def scatter_in_target(particle_location: ti.template()):  # type: ignore
            for i in particle_location:
                pixel_id = ti.random() * number_of_pixels_field[None] - threshold[None]
                for j, k in ti.ndrange(self.image_field.shape[0], self.image_field.shape[1]):
                    if self.image_field[j, k] > 0.:
                        pixel_id -= self.image_field[j, k]
                        if pixel_id <= 0.:
                            particle_location[i] = ti.Vector([j + ti.random() - 0.5, k + ti.random() - 0.5])
                            break
        return scatter_in_target

    def get_in_target_function(self, dt: float):
        if self.randomly_hit:
            @ti.func
            def in_target(x, y, do_not_consider_dt):
                result = True
                if do_not_consider_dt:
                    result = ti.random() < ti.abs(self.image_field[int(ti.round(x)), int(ti.round(y))])
                else:
                    result = ti.random() < ti.abs(self.image_field[int(ti.round(x)), int(ti.round(y))]) * dt
                return result
        else:
            @ti.func
            def in_target(x, y, do_not_consider_dt):
                return self.image_field[int(ti.round(x)), int(ti.round(y))] > 0.5
        return in_target
    
    def get_target_arrived_function(self):
        if self.remove_after_arrived:
            if self.non_negative:
                @ti.func
                def target_arrived(x, y):
                    i = int(ti.round(x))
                    j = int(ti.round(y))
                    value_change = self.image_field[i, j]
                    self.image_field[i, j] = 0.
                    return value_change
            else:
                @ti.func
                def target_arrived(x, y):
                    i = int(ti.round(x))
                    j = int(ti.round(y))
                    self.image_field[i, j] -= 1.
                    return 1.
        else:
            @ti.func
            def target_arrived(x, y):
                return 1.
        return target_arrived

class CombinedTarget(Target):
    def __init__(self, chemical_id: int, target_1: Target, target_2: Target):
        self.chemical_id = chemical_id
        target_1.chemical_id = chemical_id
        target_2.chemical_id = chemical_id
        self.target_1 = target_1
        self.target_2 = target_2

    def init_taichi_objects(self):
        self.target_1.init_taichi_objects()
        self.target_2.init_taichi_objects()

    def get_scatter_function(self):
        return None

    def get_in_target_function(self, dt: float):
        in_target_1 = self.target_1.get_in_target_function(dt)
        in_target_2 = self.target_2.get_in_target_function(dt)
        @ti.func
        def in_target(x, y, do_not_consider_dt):
            return in_target_1(x, y, do_not_consider_dt) or in_target_2(x, y, do_not_consider_dt)
        return in_target
    
    def get_target_arrived_function(self):
        target_arrived_1 = self.target_1.get_target_arrived_function()
        target_arrived_2 = self.target_2.get_target_arrived_function()
        @ti.func
        def target_arrived(x, y):
            value_change = target_arrived_1(x, y)
            value_change += target_arrived_2(x, y)
            return value_change
        return target_arrived
    