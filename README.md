![JavSP](./image/JavSPN.svg)

# Jav Scraper Package

**A Jav(Japanese Adult Video) Scraper that summarizes from multiple websites**

By grabbing the	bangou in the file name, JavSP can pull data from multiple websites and summarize them, classify them according to a predefined rule and create metadata for softwares like Emby, Jellyfin and Kodi.

**Docker & WebUI**: Due to limited time and energy, there's no Docker support yet. On top of that, User Interface is not one of the primary goal for this [project](https://github.com/Yuukiy/JavSP/issues/148). If you need Docker support, maybe you can give [JavSP-Docker](https://github.com/tetato/JavSP-Docker) a try.

![License](https://img.shields.io/github/license/Yuukiy/JavSP)
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)
[![Crawlers test](https://img.shields.io/github/actions/workflow/status/glyh/JavSPn/test-web-funcs.yml?label=crawlers%20test)](https://github.com/glyh/JavSPn/actions/workflows/test-web-funcs.yml)
[![Latest release](https://img.shields.io/github/v/release/glyh/JavSPn)](https://github.com/glyh/JavSPn/releases/latest)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

## Distinctive Features against [upstream](https://github.com/Yuukiy/JavSP)
- [x] Crawls stage photos(optional).
- [x] Support cropping cover with either: 1. face detection with yunet; 2. body segmentation with pphumanseg.
- [x] Use `cx_Freeze` to package, which is smaller in size.
- [x] Adheres to pip packaging standards, and thus can be used as a library.

## Features

This is a non-exhaustive list of implemented and unimplemented features being refined over time.

- [x] Recognize movie ID automagically
- [x] Dealing with movies separated into multiple parts
- [x] Summarize information from multiple sites to generate [NFO file](https://jellyfin.org/docs/general/server/metadata/nfo/).
- [x] Automatic tests for website crawler on a daily basis
- [x] Parallel data scraping
- [x] Downloading HD covers
- [x] AI based cover crop for atypical covers
- [x] Check new version <del>and self-updating</del>
- [x] Translating titles and descriptions
- [ ] Matching local subtitles
- [ ] Using thumb to create folder cover
- [ ] Keeping genre consistency across different websites
- [ ] Different mode of operations(Scraping and Moving, Scrape only)
- [ ] Optional: Allow user to interveine when there's a conflicts during scrapping.

## Installation

- For the impatient

	Visit [Github Release Page](https://github.com/glyh/JavSPn/releases/latest) and download the latest portable version of JavSP (Windows Only).

- Buliding from source
  - Ensure you have Python >= 3.9
  - Run the following

	```
	git clone --recurse-submodules https://github.com/glyh/JavSPn.git
	cd JavSP
	poetry install
	poetry run javspn
	```

## Usage

You can modify the configuration file `config.ini` to instruct how `JavSP` should work for you.

JavSP also accepts CLI flags and prioritize them over `config.ini`, you can run `JavSP -h` to see a list of supportted flags. 

For more detailed instructions please visit [JavSP Wiki](https://github.com/Yuukiy/JavSP/wiki)

Please file an issue if you find any problem using this software.😊 

## Bug report

If you encounter any bug that is not already encountered by other users(you can check this by searching through the issue page), don't hesitate to go and [file an isssue](https://github.com/glyh/JavSPn/issues).


## Contribution

No need to buy me any coffee LoL. If you like to help, please help me through these methods:

- Help writing and improving the Wiki

- Help completing the Unit Test (Not necessarilly coding, testcases or insightful obvervations are also welcomed)

- Help translating the genre

- Pull Request for bug fix or new feature

- Give me a star (Thank you!)


## License

This project is under the restriction of both the GPL-3.0 License and the [Anti 996 License](https://github.com/996icu/996.ICU/blob/master/LICENSE). On top of that, using this software implies that you accept the following terms: 
- I will only use this software for academic purpose

- I won't advertize this project on any Chinese social media like weibo or wechat.

- I will follow the local government regulation when using this software.

- I will not monetrize this software and make profit out of it.

