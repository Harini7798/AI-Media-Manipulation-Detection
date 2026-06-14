import os
import cv2
import json
import http.client, urllib.parse

from utils import get_filename_only

base_path = os.path.join('.', 'train_sample_videos')

# ── Credentials are loaded from environment variables. Never hardcode them. ──
#
# Export these in your shell (or put them in a local `.env` file — which is
# already gitignored — and source it) BEFORE running this script:
#
#   export AZURE_COMPUTER_VISION_NAME="xxxxxxxxxx.cognitiveservices.azure.com"
#   export AZURE_COMPUTER_VISION_API_KEY="your-real-key"
#
AZURE_COMPUTER_VISION_NAME = os.environ.get("AZURE_COMPUTER_VISION_NAME", "")
AZURE_COMPUTER_VISION_API_KEY = os.environ.get("AZURE_COMPUTER_VISION_API_KEY", "")

if not AZURE_COMPUTER_VISION_NAME or not AZURE_COMPUTER_VISION_API_KEY:
    raise SystemExit(
        "Missing Azure credentials. Set AZURE_COMPUTER_VISION_NAME and "
        "AZURE_COMPUTER_VISION_API_KEY as environment variables before running "
        "this script. Do not paste your key into source code."
    )

with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    metadata = json.load(metadata_json)
    print(len(metadata))

for filename in metadata.keys():
    tmp_path = os.path.join(base_path, get_filename_only(filename))
    print('Processing Directory: ' + tmp_path)
    frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))]
    faces_path = os.path.join(tmp_path, 'faces')
    print('Creating Directory: ' + faces_path)
    os.makedirs(faces_path, exist_ok=True)
    print('Cropping Faces from Images...')

    for frame in frame_images:
        print('Processing ', frame)
        image = cv2.cvtColor(cv2.imread(os.path.join(tmp_path, frame)), cv2.COLOR_BGR2RGB)

        # Open the binary file
        with open(os.path.join(tmp_path, frame), 'rb') as file_contents:
            img_data = file_contents.read()

        ######### Azure Computer Vision API
        headers = {
            # Request headers
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': AZURE_COMPUTER_VISION_API_KEY,
        }

        params = urllib.parse.urlencode({
            # Request parameters
            'visualFeatures': 'Faces'
        })

        try:
            conn = http.client.HTTPSConnection(AZURE_COMPUTER_VISION_NAME)
            conn.request("POST", "/vision/v3.0/analyze?%s" % params, img_data, headers)
            response = conn.getresponse().read()
            data = json.loads(response.decode('utf-8'))
            print(data)
            conn.close()
        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))
            continue
        
        print(data['faces'])
        print('Face Detected: ', len(data['faces']))
        count = 0

        for result in data['faces']:
            bounding_box = []
            bounding_box.append(result['faceRectangle']['left'])
            bounding_box.append(result['faceRectangle']['top'])
            bounding_box.append(result['faceRectangle']['width'])
            bounding_box.append(result['faceRectangle']['height'])
            print(bounding_box)

            margin_x = bounding_box[2] * 0.3  # 30% as the margin
            margin_y = bounding_box[3] * 0.3  # 30% as the margin
            x1 = int(bounding_box[0] - margin_x)
            if x1 < 0:
                x1 = 0
            x2 = int(bounding_box[0] + bounding_box[2] + margin_x)
            if x2 > image.shape[1]:
                x2 = image.shape[1]
            y1 = int(bounding_box[1] - margin_y)
            if y1 < 0:
                y1 = 0
            y2 = int(bounding_box[1] + bounding_box[3] + margin_y)
            if y2 > image.shape[0]:
                y2 = image.shape[0]
            print(x1, y1, x2, y2)
            crop_image = image[y1:y2, x1:x2]
            new_filename = '{}-{:02d}.png'.format(os.path.join(faces_path, get_filename_only(frame)), count)
            count = count + 1
            cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
