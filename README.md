# GoogleFaceScraper
A Google image scraper that to collect a clean labeled dataset. You can either use a text file containing names of use the integrated IMDB name scraper to make the script fully automatic.

In the output folder a text file will be created, containing links to all scraped images and the date they were scraped.

For each person a sub folder `name_lastname` will be made in the chosen output folder. The faces beloning to the person will be saved in the subfolder as:
```
name_lastname
	name_lastname_0001.png
	name_lastname_0002.png
	name_lastname_0003.png
	...
```

**Important:** if you use your own text file with names make sure the names are formatted as **name_lastname**.

## Getting started
You can follow the instructions below to deploy this project to your local machine.

### Prerequisites
For this project to work you first need to install some dependencies. Most dependencies can be installed using `pip install -r dependencies.txt`. But you will need to install these dependencies manually:

* [Facenet](https://github.com/davidsandberg/facenet)
* CUDA
* cuDNN
* Tensorflow-gpu 1.0
* OpenCV 3

### Install
Follow the steps below to install and run the project:
1. Clone this repository `$ git clone https://github.com/MaartenBloemen/GoogleFaceScraper.git`
2. Run the scraper `$ python run src/scraper.py /path/to/model.pb /path/to/output/dir/`
	Optional arguments:
	*  `--name_source` - **String** - Path to txt file, don't use if you want to use the IMDB scraper
	*  `--limit` - **int** - Number of IMDB name pages to use (default: 100), number of people are the chosen limit * 50
	*  `--image_size` - **int** - The width and the height the images will be saved as (default: 160).
	* `--margin` - **int** - The margin around the egde of the face and egde of the image (default: 44).
	* `--min_cluster_size` - **int** - The minimum amount of pictures required for a single cluster (default: 10), note only the largest cluster will be safed.
	* `--cluster_threshold` - **float** - The minimum ecleudian distance for an image to be part of a cluster (default: 1.0).
	* `--safe_mode` - **String** - Choices ['on', 'off], this determines whether the Google search should include explicit images or not (default: on).
	* `--gpu_memory_fraction` - **float** - A number bewteen 0 and 1 that determines that max percentage of GPU memory that can be used (default: 1.0).
	
You can find a pretrained model (.pb) on the [Facenet](https://github.com/davidsandberg/facenet) repository as well as instructions on how to train your own model.

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/MaartenBloemen/GoogleFaceScraper/blob/master/LICENSE.md)  file for details.

## Authors
* [Maarten Bloemen](https://github.com/MaartenBloemen) 

## Acknowledgments
* [Facenet](https://github.com/davidsandberg/facenet) - MIT License
* [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment) - MIT License

## Disclaimer
This project is meant as a proof of concept. Use datasets gathered this way ONLY for research purposes.
