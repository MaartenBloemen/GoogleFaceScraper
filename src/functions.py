# MIT License
#
# Copyright (c) 2017 Maarten Bloemen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import urllib2
import cv2
import json
import numpy as np
from bs4 import BeautifulSoup


class GoogleFunctions:
    def __init__(self):
        pass

    def _get_soup(self, url, header):
        return BeautifulSoup(urllib2.urlopen(urllib2.Request(url, headers=header)), 'html.parser')

    def _get_image(self, url):
        try:
            req = urllib2.urlopen(url)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            return None

    def _get_image_links(self, query, safe_mode):
        query = query.strip().split('_')
        query = '+'.join(query)
        url = 'https://www.google.com/search?q=' + query + '&source=lnms&tbm=isch&safe=' + safe_mode
        print(url)
        header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36'
        }
        soup = self._get_soup(url, header)

        images_links = []  # contains the link for Large original images, type of  image
        for a in soup.find_all('div', {'class': 'rg_meta'}):
            link, Type = json.loads(a.text)['ou'], json.loads(a.text)['ity']
            images_links.append((link, Type))

        return images_links

    def get_images(self, query, safe_mode):
        image_links = self._get_image_links(query, safe_mode)
        images = []
        for i, (img, Type) in enumerate(image_links):
            image = self._get_image(img)
            if image is not None:
                images.append(image)

        return images


class ImdbFunctions:
    def __init__(self):
        pass

    def _is_ascii(self, name):
        return all(ord(char) < 128 for char in name)

    def _get_soup(self, url, header):
        return BeautifulSoup(urllib2.urlopen(urllib2.Request(url, headers=header)), 'html.parser')

    def get_celebrity_names(self, limit):
        celebrity_list = []
        for i in range(limit):
            start = 1 + (i * 50)
            url = 'http://www.imdb.com/search/name?gender=male,female&start=' + str(start) + '&ref_=rlm'
            header = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36'
            }
            soup = self._get_soup(url, header)

            check = soup.findAll('h1')
            if 'Error' in check:
                break

            names = soup.findAll('td', {'class': 'name'})
            for name in names:
                name = name.a.string.split()
                name = '_'.join(name)
                if self._is_ascii(name):
                    celebrity_list.append(name)

        return celebrity_list
