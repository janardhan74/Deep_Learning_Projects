from flask import Flask,render_template,request,redirect
import tensorflow
import numpy as np

app = Flask(__name__)



def breast_cancer_prediction(x):
  '''
  x -> [['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']]
  y -> target
      0 -> malignant
      1 -> benign
  '''
  model = tensorflow.keras.models.load_model('model/breast_cancer_prediction.h5')
  return "You have MALIGNANT BREAST CANCER " if np.argmax(model.predict(x)) == 0 else "You have BENIGN BREAST CANCER"

@app.route('/breast_cancer_prediction',methods=['POST','GET'])
def index():
    features_obj = [('mean_radius', None), ('mean_texture', None), ('mean_perimeter', None), ('mean_area', None), ('mean_smoothness', None), ('mean_compactness', None), ('mean_concavity', None), ('mean_concave_points', None), ('mean_symmetry', None), ('mean_fractal_dimension', None), ('radius_error', None), ('texture_error', None), ('perimeter_error', None), ('area_error', None), ('smoothness_error', None), ('compactness_error', None), ('concavity_error', None), ('concave_points_error', None), ('symmetry_error', None), ('fractal_dimension_error', None), ('worst_radius', None), ('worst_texture', None), ('worst_perimeter', None), ('worst_area', None), ('worst_smoothness', None), ('worst_compactness', None), ('worst_concavity', None), ('worst_concave_points', None), ('worst_symmetry', None), ('worst_fractal_dimension', None)]
    if request.method == 'POST':
        form_obj = dict(request.form)
        test_val = []
        for feature,val in form_obj.items():
            test_val.append(float(val))
        test_val = np.array([test_val])
        prediction = breast_cancer_prediction(test_val)
        features_obj = form_obj.items()
        print(prediction)
        return render_template('index.html',prediction=prediction,features_obj=features_obj)
    else:
        return render_template('index.html',features_obj=features_obj,prediction=None)
 

if __name__ == '__main__':
    app.run(debug=True)