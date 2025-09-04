
![Google Summer of Code logo](https://summerofcode.withgoogle.com/assets/media/logo.svg)

# GSoC 2025 

Name:  [Dimitrios Athanasopoulos](https://github.com/jimmmyss)

Mentor:  [Foivos Karounos](https://github.com/fffoivos), [Nikos Tsekos](https://github.com/nikostsekos)

Project Website:  [https://glossapi.gr](https://glossapi.gr)

Project GitHub:  [https://github.com/eellak/glossAPI](https://github.com/eellak/glossAPI)

Project HuggingFace:  [https://huggingface.co/glossAPI](https://huggingface.co/glossAPI)

GSoC Page:  [https://summerofcode.withgoogle.com/programs/2025/projects/WaioHmfG](https://summerofcode.withgoogle.com/programs/2025/projects/WaioHmfG)

GSoC GitHub:  [https://github.com/jimmmyss/GSoC-2025](https://github.com/jimmmyss/GSoC-2025)


##  Abstract

This project addresses the lack of accessible, Greek high-quality datasets for training Large Language Models (LLMs). Although robust LLM frameworks are widely available, open-source Greek-specific datasets are severely underrepresented, especially those that captures regional variations and dialects. This project's mission is to integrate Optical Character Recognition (OCR) capabilities into the processing pipeline and expand its dataset resources to support these linguistic nuances.

## Work and Repository

The scope of this project was to analyze large pre-crawled datasets from Common Crawl, study their pipelines and entries, extract data that could serve as a valuable addition to the existing datasets, and incorporate OCR capabilities into the GlossAPI pipeline.

During the time of the program, i also played a key role in the development of [glossapi.gr](https://glossapi.gr) site and discord server, including the development of a discord bot that is responsible for announcing and moderating text channels

## Deliverables

### Dataset
- Analyzed [Oscar](https://oscar-project.org/) and [HPLT](https://hplt-project.org/datasets/v2.0) datasets (Approx. 700GB).
- Scraped, extracted and processed [Greek](https://search.et.gr/) and [European](https://eur-lex.europa.eu/) legislations (Approx. 3GB).
- Scraped, extracted and processed [OpenArchives](https://www.openarchives.gr/) and [OpenBooks](https://www.openbook.gr/) (Approx. GB).

### OCR
- Created and tested multiple OCR pipelines using [Tesseract](https://github.com/tesseract-ocr/tesseract) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for simple text extraction.
- Evaluated [Qwen-VL](https://github.com/QwenLM/Qwen-VL) Vision-Language Model (VLM) pipeline for complex text extraction.
- Developed a custom post-processing pipeline for spell and grammar checking using [Hunspell](https://github.com/hunspell/hunspell).

### Discord
- Created a [Discord Community Server](https://discord.com/invite/TY69npdMwM).
- Developed a custom Discord bot that is responsible for announcing and moderating text channels.

### Site
- Contributed in the development of the official [GlossAPI landing page](https://glossapi.gr).

## Key Takeaways

### Dataset

The analysis started with the idea of classifying and categorizing each domain based on the type of the site it was. We followed two different approaches to calculate the most effective one:

**1. Classification through metadata**

This step required further analysis of the whole datasets and link aggregation based on the same domain. After creating a parquet that had all the domain entries and how many times a domain was inside the dataset, we had to scrape the following metadata for each domain:

1. Redirect URL.
2. HTTP `status_code` returned by the server.  
3. `title`  
4. `description`  
5. `keywords`  
6. Open Graph Metadata  
   - `og_title`  
   - `og_description`  
   - `og_type`  

This process proved to be unoptimal due to the fact that scraping had to be done to almost 1.000.000 different domains from both datasets. Doing it in batches momenteraly solved the problem of getting stuck in this step and not being able to do further analysis, but after it was done for OSCAR, which was the smaller dataset, there seemed to be a large gap in the actual metadata. More specifically our analysis showed:

- The percentage of missing metadata for all the domains:

| Field             | Missing (%) |
|-------------------|-------------|
| title             | 16%         |
| meta_description  | 44%         |
| keywords          | 83%         |
| og_title          | 49%         |
| og_description    | 58%         |
| og_type           | 53%         |

- The percentage of domains that have a matching `title` & `og_title` and `description` & `og_description`:

| Field Comparison                 | Match Rate |
|----------------------------------|------------|
| title & og_title                 | 74.4%      |
| description & og_description     | 88.0%      |

- The percentage of domains that have `og_title` but not `title` and `og_description` but no `description`:

| Missing Field                            | Percentage |
|------------------------------------------|------------|
| title & og_title                         | 0.6%       |
| description & og_description             | 0.0%       |

The next step was to assign each domain to one of our predefined categories by analyzing either its metadata or its landing‐page content. For metadata‐based classification, we employed two approaches: an ensemble of six embedding models that each voted on the best category, and a separate pipeline powered by Gemini. For landing‐page classification, we fed the full HTML content into Gemini, which then determined the most appropriate category based on the page’s actual structure and text.

**2. Classification through landing page content:**

This step begins by scanning all of the URLs and page contents from the datasets and then to filter out anything that isnt a landing page. After isolating each domain’s primary page and its full HTML content we load this curated dataset and apply  BERTopic analysis to generate and discover new concise, human-readable category labels. The results can be used to classify each domain. 

**Conclusion**
- The categorization with local embedding models, despite being fast proved to be unreliable, especially in domains that lacked metadata.
- The categorization with Gemini, proved to be more reliable, but it can only be used for reference because of its API limits.
- The BERTopic analysis proved to be "nonsensical" because the landing page content lacked coherent subject matter.
- The Common Crawl extraction was discontinued after we found no high-quality crawls for the filtered-out entries.
- The analysis showed that it is more beneficial to focus on academic papers and open books, since open web data is already widely available and well-covered.

### OCR

The objective was to find the most accurate OCR solution for processing Greek text, that will then be incorporated inside of the GlossAPI pipeline.

**1. TesseractOCR**

The initial testing scores showed on average an 70-75% total accuracy score with most notable mistakes occuring in complex text contents such as mathematical formulas, tables, and structured documents. Errors also appeared in simpler text, particularly with unusual polytonic accents or occasional incorrect characters in words.

To address this, a custom post-processing pipeline using Hunspell was developed. This pipeline first strips all accents from the text, then applies Greek spell checking to fix incorrect characters and reapply the correct accents. Additionally, the Tesseract model was trained using in-house data from scrapped PDFs, consisting of paired image files (.tif) and their corresponding ground-truth text files (.gt.txt). With this approach, accuracy improved to over 90% for standard text, though complex content such as mathematical formulas and structured tables remained challenging.

**2. PaddleOCR** 

The initial testing showed a similar average accuracy of 65-70%, with frequent misrecognition of accented characters and polytonic text. While paragraph structure and basic Greek characters were generally preserved, there were instances where entire lines were completely garbled or nonsensical. Compared to Tesseract, PaddleOCR handled noisy or varied fonts slightly better, but it struggled to maintain character-level accuracy and proper accents. Additionally, PaddleOCR is more resource-intensive than Tesseract, as it relies on deep learning models that require more memory and processing power, and it is generally slower on CPU, though GPU inference can speed up batch processing.

To address common recognition errors, a custom post-processing pipeline similar to the one used for Tesseract was developed. This pipeline strips all accents from the text, applies Greek spell checking using Hunspell to correct incorrect characters and reapply the correct accents. Unlike Tesseract, no additional custom training was performed for PaddleOCR at this stage. With this approach, accuracy improved for standard text to 70-75%, though there were still instances of garbled lines. Further training is required to optimize performance for Greek text and effectively eliminate these garbled lines.

**3. Qwen-VL**

The results of the evaluation showed excellent performance on both simple and complex documents, achieving accuracy scores of up to 98% in many instances. Unlike Tesseract and PaddleOCR, Qwen-VL was able to process mathematical formulas, structured tables, and dense academic content with very high reliability, while also being capable of producing LaTeX representations of equations, making it significantly more effective than the other OCR pipelines.

The main drawback is its computational cost. While Tesseract and PaddleOCR can run efficiently on CPUs or lightweight GPUs (<1 GB VRAM), Qwen-VL requires a modern high-memory GPU (15–40 GB VRAM) and noticeably longer processing times. As GlossAPI is intended for accessibility by the average user rather than being limited to power users with high-end hardware only it was chosen not to be integrated into the GlossAPI pipeline.

**Conclusion**

After evaluating all approaches, the project selected a combination of Docling with RapidOCR (ONNX, Greek) as the optimal solution for GlossAPI’s OCR requirements. Since RapidOCR is built on top of PaddleOCR, which demonstrated strong potential when trained, it was concluded to be the most promising option for long-term adoption.


## Future Work
The project still has several unresolved issues and potential improvements related to my work. 
I plan to continue addressing these by working on my open pull requests and assisting in maintaining the project in the future.

This is the future work.
1.
2.
3.

## Challenges
Working on the platform this summer was one of the year's best moments.

During the project, I encountered several challenges and learned many important things:
-   I was exposed to new methodologies and technologies that expanded my existing knowledge.
-   I faced numerous bugs and learned various methods for troubleshooting and resolving issues.
-   I collaborated with a large organization, which provided valuable experience.
-   I gained experience with AI practices that I had not previously encountered.

I am also very pleased with the challenges I faced, as they contributed significantly to my growth and improvement. Moreover, working on this project has been particularly rewarding because contributing to education brings an extra sense of fulfillment and joy.

## Thanks note
This project is supported by the Google Summer of Code program and the Open Technologies Alliance - Gfoss organization. I would like to extend my sincere thanks to my mentors, Foivos and Nikos, for his invaluable guidance throughout the project, covering areas such as implementation, code review, and community involvement. Additionally, I am grateful to the organization for providing me with the opportunity to work on this project. It was an absolute pleasure working with you and I will try to stay connected with the community in the future!

## Conclusion
In conclusion, the development of the GlossAPI as part of Google Summer of Code 2025 has been an enriching and transformative experience. The project successfully introduced a range of advanced features and enhancements designed to improve the educational experience for users. The integration of interactive simulations, improved code, and a more intuitive user interface has significantly contributed to the platform's functionality and user experience.

Throughout the project, I have gained valuable insights into new technologies and methodologies, honed my problem-solving skills, and expanded my expertise in AI practices. Collaborating with a large organization and engaging with the broader community has been incredibly rewarding, both professionally and personally.

As the project continues to evolve, I remain committed to addressing outstanding issues and implementing further improvements. The journey has been fulfilling, and the opportunity to contribute to educational technology has been particularly gratifying. I look forward to staying connected with the community and continuing to support the platform's development in the future.

Thank you once again to everyone who supported and guided me throughout this project. Your contributions and encouragement have made this experience truly memorable.
