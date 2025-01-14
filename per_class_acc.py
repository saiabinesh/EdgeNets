# Per-class accuracies for 40-class model
accuracies_40 = {
    'apple': 92, 'aquarium_fish': 80, 'baby': 75, 'bee': 74, 'beetle': 76, 'bottle': 86,
    'bridge': 74, 'bus': 67, 'butterfly': 75, 'castle': 81, 'chair': 87, 'crab': 69,
    'dolphin': 80, 'forest': 59, 'fox': 76, 'house': 69, 'kangaroo': 58, 'lion': 78,
    'lobster': 65, 'maple_tree': 74, 'motorcycle': 92, 'mountain': 85, 'mouse': 48,
    'mushroom': 75, 'palm_tree': 85, 'plate': 85, 'poppy': 92, 'possum': 52, 'ray': 65,
    'road': 91, 'shrew': 56, 'skunk': 93, 'skyscraper': 88, 'squirrel': 50, 'streetcar': 64,
    'table': 65, 'television': 86, 'tractor': 84, 'willow_tree': 61, 'worm': 80
}

# Per-class accuracies for 70-class model
accuracies_70 = {
    'apple': 83, 'baby': 55, 'bear': 54, 'beaver': 52, 'bee': 78, 'beetle': 74,
    'bicycle': 83, 'bottle': 79, 'boy': 51, 'bridge': 81, 'bus': 58, 'caterpillar': 66,
    'cattle': 69, 'chair': 82, 'clock': 69, 'cockroach': 83, 'couch': 61, 'crab': 56,
    'dinosaur': 66, 'dolphin': 73, 'flatfish': 68, 'forest': 62, 'girl': 36, 'hamster': 70,
    'keyboard': 82, 'lamp': 64, 'lawn_mower': 84, 'leopard': 69, 'lion': 76, 'lizard': 48,
    'lobster': 52, 'man': 52, 'maple_tree': 64, 'mountain': 90, 'mouse': 55, 'mushroom': 68,
    'oak_tree': 72, 'orange': 91, 'orchid': 79, 'otter': 46, 'palm_tree': 85, 'pear': 74,
    'pickup_truck': 83, 'plain': 94, 'plate': 76, 'poppy': 76, 'porcupine': 64, 'rabbit': 56,
    'raccoon': 81, 'ray': 66, 'rocket': 79, 'rose': 61, 'seal': 47, 'skyscraper': 89,
    'snail': 67, 'snake': 60, 'streetcar': 76, 'sweet_pepper': 59, 'table': 63, 'telephone': 73,
    'tiger': 72, 'tractor': 77, 'train': 74, 'trout': 79, 'tulip': 70, 'turtle': 49,
    'wardrobe': 93, 'willow_tree': 60, 'woman': 44, 'worm': 70
}

# Per-class accuracies for 100-class model
accuracies_100 = {
    'apple': 86, 'aquarium_fish': 74, 'baby': 57, 'bear': 48, 'beaver': 51, 'bed': 70,
    'bee': 69, 'beetle': 63, 'bicycle': 78, 'bottle': 78, 'bowl': 47, 'boy': 49,
    'bridge': 79, 'bus': 53, 'butterfly': 64, 'camel': 71, 'can': 63, 'castle': 76,
    'caterpillar': 65, 'cattle': 61, 'chair': 82, 'chimpanzee': 85, 'clock': 64,
    'cloud': 74, 'cockroach': 79, 'couch': 53, 'crab': 58, 'crocodile': 50, 'cup': 73,
    'dinosaur': 61, 'dolphin': 47, 'elephant': 66, 'flatfish': 65, 'forest': 60, 'fox': 72,
    'girl': 37, 'hamster': 70, 'house': 65, 'kangaroo': 54, 'keyboard': 82, 'lamp': 60,
    'lawn_mower': 80, 'leopard': 63, 'lion': 76, 'lizard': 38, 'lobster': 53, 'man': 40,
    'maple_tree': 66, 'motorcycle': 91, 'mountain': 79, 'mouse': 43, 'mushroom': 73,
    'oak_tree': 72, 'orange': 89, 'orchid': 72, 'otter': 38, 'palm_tree': 79, 'pear': 73,
    'pickup_truck': 80, 'pine_tree': 59, 'plain': 80, 'plate': 71, 'poppy': 72,
    'porcupine': 52, 'possum': 50, 'rabbit': 46, 'raccoon': 77, 'ray': 50, 'road': 91,
    'rocket': 72, 'rose': 65, 'sea': 77, 'seal': 42, 'shark': 58, 'shrew': 50,
    'skunk': 87, 'skyscraper': 85, 'snail': 52, 'snake': 59, 'spider': 68, 'squirrel': 43,
    'streetcar': 71, 'sunflower': 89, 'sweet_pepper': 61, 'table': 57, 'tank': 81,
    'telephone': 65, 'television': 79, 'tiger': 68, 'tractor': 80, 'train': 76,
    'trout': 82, 'tulip': 50, 'turtle': 40, 'wardrobe': 90, 'whale': 68, 'willow_tree': 53,
    'wolf': 71, 'woman': 45, 'worm': 75
}

# Find common classes
common_classes = set(accuracies_40.keys()) & set(accuracies_70.keys()) & set(accuracies_100.keys())

# Compare accuracy across models for each common class
print(f"{'Class':<15} {'40-Class':<10} {'70-Class':<10} {'100-Class':<10}")
for cls in common_classes:
    acc_40 = accuracies_40.get(cls, 0)
    acc_70 = accuracies_70.get(cls, 0)
    acc_100 = accuracies_100.get(cls, 0)
    print(f"{cls:<15} {acc_40:<10} {acc_70:<10} {acc_100:<10}")

# Average accuracies for common classes
avg_acc_40 = sum(accuracies_40[cls] for cls in common_classes) / len(common_classes)
avg_acc_70 = sum(accuracies_70[cls] for cls in common_classes) / len(common_classes)
avg_acc_100 = sum(accuracies_100[cls] for cls in common_classes) / len(common_classes)

print(f"\nAverage Accuracy for Common Classes:")
print(f"40-Class Model: {avg_acc_40:.2f}%")
print(f"70-Class Model: {avg_acc_70:.2f}%")
print(f"100-Class Model: {avg_acc_100:.2f}%")
