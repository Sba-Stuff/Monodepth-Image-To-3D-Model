# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from __future__ import absolute_import, division, print_function
from flask import *
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms, datasets
import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR
from werkzeug import secure_filename

#TEMPLATE_DIR = os.path.abspath('../templates')
#STATIC_DIR = os.path.abspath('../static')
# app = Flask(__name__) # to make the app run without any
#app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {"jpg"}
app = Flask(__name__,template_folder="template/",static_folder="static/")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#@app.route('/output/<path:filepath>')
#def stater(filepath):
#    return send_from_directory('output', filepath)

def DoWork(default_model):
	model_name=default_model
	download_model_if_doesnt_exist(model_name)
	model_path = os.path.join("models", model_name)
	print("-> Loading model from ", model_path)
	encoder_path = os.path.join(model_path, "encoder.pth")
	depth_decoder_path = os.path.join(model_path, "depth.pth")
	print("   Loading pretrained encoder")
	encoder = networks.ResnetEncoder(18, False)
	device = torch.device("cpu")
	loaded_dict_enc = torch.load(encoder_path, map_location=device)

	# extract the height and width of image that this model was trained with
	feed_height = loaded_dict_enc['height']
	feed_width = loaded_dict_enc['width']
	filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
	encoder.load_state_dict(filtered_dict_enc)
	encoder.to(device)
	encoder.eval()

	print("   Loading pretrained decoder")
	depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
	loaded_dict = torch.load(depth_decoder_path, map_location=device)
	depth_decoder.load_state_dict(loaded_dict)
	depth_decoder.to(device)
	depth_decoder.eval()
	path = "static/input.jpg"
	output_directory = os.path.dirname("static/")
	print("-> Predicting on {:d} test images".format(len(path)))
	# PREDICTING ON EACH IMAGE IN TURN
	with torch.no_grad():
	            # Load image and preprocess
	    input_image = pil.open(path).convert('RGB')
	    original_width, original_height = input_image.size
	    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
	    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

	            # PREDICTION
	    input_image = input_image.to(device)
	    features = encoder(input_image)
	    outputs = depth_decoder(features)

	    disp = outputs[("disp", 0)]
	    disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)

	            # Saving colormapped depth image
	    disp_resized_np = disp_resized.squeeze().cpu().numpy()
	    vmax = np.percentile(disp_resized_np, 95)
	    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
	    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
	    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
	    im = pil.fromarray(colormapped_im)
	    output_name = os.path.splitext(os.path.basename(path))[0]
	    name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
	    im.save(name_dest_im)

	    print("   Processed image")
	    print("   - {}".format(name_dest_im))
	print('-> Done!')

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def upload_file():
    fil = []
    if request.method == 'POST':
        f = request.files['file']
        if allowed_file(f.filename):
            f.save(UPLOAD_FOLDER+secure_filename("input.jpg"))
            fil.append("File uploaded successfully")
            fil.append(UPLOAD_FOLDER+secure_filename(f.filename))
            return fil
        else:
            fil.append("Not An Expected File. Only JPG images are accepted.")
            return fil
@app.route('/',methods=['GET', 'POST'])
def hello_world():
    legoutput = upload_file()
    lig = "This is a demo utterance. This will work when you do not add any utterance."
    #return mainpage()
    if str(legoutput)=="None":
        return render_template("index.html",output="You Know What To Do, Upload Image To Get Depth")
    else:
        try:
            model_name = request.form["models"]
            print(model_name)
            DoWork(model_name)
            codehtml = "<b>Original:</b></br><img src='static/input.jpg' height='293' width='453'></br>Depth:</b></br><img src='static/input_disp.jpeg' height='293' width='453'></br>"
            return render_template("index.html",output=codehtml)
        except Exception as e:
            return render_template("index.html",output="Caught exception: %s" % repr(e))
    #return xieon
# main driver function
@app.route('/3DModel/',methods=['GET', 'POST'])
def ThreeD_Model():
    return render_template("index2.html")

if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run()