class NaiveBayesClassifier:
    def __init__(self, dataset):
        self.dataset = dataset
        self.prior_probs = self.calculate_prior_probabilities()
    
    def calculate_prior_probabilities(self):
        class_counts = {}
        total_samples = len(self.dataset)
        
        for data in self.dataset:
            label = data['play']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        return {cls: count / total_samples for cls, count in class_counts.items()}
    
    def calculate_likelihood(self, feature, value, given_class):
        count_feature_given_class = sum(1 for data in self.dataset if data[feature] == value and data['play'] == given_class)
        count_class = sum(1 for data in self.dataset if data['play'] == given_class)
        
        return (count_feature_given_class + 1) / (count_class + len(set(data[feature] for data in self.dataset)))  # Laplace smoothing
    
    def predict(self, test_sample):
        classes = self.prior_probs.keys()
        probabilities = {}
        
        for cls in classes:
            probabilities[cls] = self.prior_probs[cls]  # Using direct probability instead of log
            
            for feature, value in test_sample.items():
                if feature != 'play':  # Ignore the class label
                    probabilities[cls] *= self.calculate_likelihood(feature, value, cls)
        
        return max(probabilities, key=probabilities.get)  # Return class with highest probability

# Sample dataset (categorical features)
dataset = [
    {'weather': 'Sunny', 'temperature': 'Hot', 'play': 'No'},
    {'weather': 'Sunny', 'temperature': 'Hot', 'play': 'No'},
    {'weather': 'Overcast', 'temperature': 'Hot', 'play': 'Yes'},
    {'weather': 'Rainy', 'temperature': 'Mild', 'play': 'Yes'},
    {'weather': 'Rainy', 'temperature': 'Cool', 'play': 'Yes'},
    {'weather': 'Rainy', 'temperature': 'Cool', 'play': 'No'},
    {'weather': 'Overcast', 'temperature': 'Cool', 'play': 'Yes'},
    {'weather': 'Sunny', 'temperature': 'Mild', 'play': 'No'},
    {'weather': 'Sunny', 'temperature': 'Cool', 'play': 'Yes'},
    {'weather': 'Rainy', 'temperature': 'Mild', 'play': 'Yes'},
    {'weather': 'Sunny', 'temperature': 'Mild', 'play': 'Yes'},
    {'weather': 'Overcast', 'temperature': 'Mild', 'play': 'Yes'},
    {'weather': 'Overcast', 'temperature': 'Hot', 'play': 'Yes'},
    {'weather': 'Rainy', 'temperature': 'Mild', 'play': 'No'}
]

# User input with validation
valid_weathers = {'Sunny', 'Overcast', 'Rainy'}
valid_temperatures = {'Hot', 'Mild', 'Cool'}

while True:
    try:
        test_weather = input("Enter weather (Sunny, Overcast, Rainy): ").strip()
        if test_weather not in valid_weathers:
            raise ValueError("Invalid input! Please enter a valid weather option: Sunny, Overcast, or Rainy.")
        
        test_temperature = input("Enter temperature (Hot, Mild, Cool): ").strip()
        if test_temperature not in valid_temperatures:
            raise ValueError("Invalid input! Please enter a valid temperature option: Hot, Mild, or Cool.")
        
        break
    except ValueError as e:
        print(e)

# Prediction
classifier = NaiveBayesClassifier(dataset)
test_sample = {'weather': test_weather, 'temperature': test_temperature}
predicted_class = classifier.predict(test_sample)
print(f"Predicted class for {test_sample}: {predicted_class}")
