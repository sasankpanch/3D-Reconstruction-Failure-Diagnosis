"""UAVScenes semantic classes (from the official cmap.py, read as data; no cmap.py in the download).
These are CLASS labels (what a surface is), not error labels and not failure causes."""
CMAP = {0: "background", 1: "roof", 2: "dirt_motor_road", 3: "paved_motor_road", 4: "river", 5: "pool",
        6: "bridge", 9: "container", 10: "airstrip", 11: "traffic_barrier", 13: "green_field", 14: "wild_field",
        15: "solar_board", 16: "umbrella", 17: "transparent_roof", 18: "car_park", 19: "paved_walk",
        20: "sedan", 24: "truck"}
CRGB = {0: (0, 0, 0), 1: (119, 11, 32), 2: (180, 165, 180), 3: (128, 64, 128), 4: (173, 216, 230), 5: (0, 80, 100),
        6: (150, 100, 100), 9: (250, 170, 30), 10: (81, 0, 81), 11: (102, 102, 156), 13: (107, 142, 35),
        14: (210, 180, 140), 15: (220, 220, 0), 16: (153, 153, 153), 17: (0, 0, 90), 18: (250, 170, 160),
        19: (244, 35, 232), 20: (0, 0, 142), 24: (0, 0, 70)}
NO_CLASS = 255

def name(i: int) -> str:
    return CMAP.get(int(i), f"id{int(i)}")
