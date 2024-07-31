from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

def my_func():
    # create a checker pattern
    checker = Checker(100, 10)
    checker.draw()
    checker.show()
    # create a circle
    circle = Circle(1024, 200, (512, 256))
    circle.draw()
    circle.show()
    spectrum = Spectrum(100)
    spectrum.draw()
    spectrum.show()

if __name__ == '__main__':
    my_func()


def img_gen():
    label_path = 'src_to_implement/Labels.json'
    file_path = 'src_to_implement/exercise_data'
    gen = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=True, mirroring=False,
                             shuffle=False)
    gen.show()
    gen2 = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=True,
                             shuffle=False)
    gen2.show()

    batch = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=False,
                                shuffle=False)
    batch.show()


if __name__ == '__main__':
    img_gen()