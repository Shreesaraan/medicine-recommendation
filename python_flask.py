import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the medicine data dictionary from 'medicine_dict.pkl'
with open('medicine_dict.pkl', 'rb') as data_file:
    medicine_dict = pickle.load(data_file)

# Load the similarity matrix from 'similarity.pkl'
with open('similarity.pkl', 'rb') as similarity_file:
    similarity = pickle.load(similarity_file)

@app.route('/')
def index():
    return render_template('index.html')

def model_recommendation(medicine_name, medicine_dict, similarity):
    # Get the index of the selected medicine
    selected_medicine_index = None
    for index, name in medicine_dict['Drug_Name'].items():
        if name == medicine_name:
            selected_medicine_index = index
            break

    if selected_medicine_index is not None:
        # Calculate similarities and recommend top medicines
        similarities = similarity[selected_medicine_index]
        recommendations = []

        # Sort recommendations by similarity and take the top 5
        recommended_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[1:6]

        for index in recommended_indices:
            recommendations.append(medicine_dict['Drug_Name'][index])

        return recommendations

    else:
        return ["Medicine not found in the dataset"]



@app.route('/recommend', methods=['POST'])
def recommend():
    medicine_name = request.form['medicine_name']

    # Perform the recommendation using the loaded data
    recommendations = model_recommendation(medicine_name, medicine_dict, similarity)

    return render_template('results.html', medicine_name=medicine_name, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
