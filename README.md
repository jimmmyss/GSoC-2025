
![Google Summer of Code logo](https://summerofcode.withgoogle.com/assets/media/logo.svg)

# GSoC 2025 

Name:  [Dimitrios Athanasopoulos](https://github.com/jimmmyss)

Mentor:  [Foivos Karounos](https://github.com/fffoivos), [Nikos Tsekos](https://github.com/nikostsekos)

Project Website:  [https://glossapi.gr](https://glossapi.gr)

Project GitHub:  [https://github.com/eellak/glossAPI](https://github.com/eellak/glossAPI)

Project HuggingFace:  [https://huggingface.co/glossAPI](https://huggingface.co/glossAPI)

GSoC Page:  [https://summerofcode.withgoogle.com/programs/2025/projects/WaioHmfG](https://summerofcode.withgoogle.com/programs/2025/projects/WaioHmfG)

##  Abstract

This project addresses the lack of accessible, Greek high-quality datasets for training Large Language Models (LLMs). Although robust LLM frameworks are widely available, open-source Greek-specific datasets are severely underrepresented, especially those that captures regional variations and dialects. This project's mission is to integrate Optical Character Recognition (OCR) capabilities into the processing pipeline and expand its dataset resources to support these linguistic nuances.

A significant part of the project was experimental and evaluative. The work focused on assessing different datasets, processing methods, OCR and VLM engines, to try to understand which tools were accurate, scalable, practical and logical enough to be integrated into GlossAPI.

## Work and Repository

The scope of this project was first to analyze large pre-crawled Common Crawl datasets and extract data that could serve as a valuable addition to the existing datasets, and second to evaluate and incorporate OCR and VLM capabilities into GlossAPI.

During the time of the program, i also played a key role in the scrapping and processing of 5 major repositories and archives and converting the collected data into structured Parquet datasets, and the development of [glossapi.gr](https://glossapi.gr) site and [discord server](https://discord.com/invite/TY69npdMwM), together with a discord bot that is responsible for announcing and moderating text channels.

## Deliverables

### Dataset
- Analyzed and classified domain specific entries of [Oscar](https://oscar-project.org/) and [HPLT](https://hplt-project.org/datasets/v2.0) datasets.
- Scraped and processed [Greek](https://search.et.gr/), [European](https://eur-lex.europa.eu/) legislations, [OpenArchives](https://www.openarchives.gr/), [OpenBooks](https://www.openbook.gr/) & [Internet Archive](https://archive.org/) (Approx 1.5TB).

### OCR
- Evaluated locally trained [Tesseract](https://github.com/tesseract-ocr/tesseract), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and [RapidOCR](https://github.com/RapidAI/RapidOCR) for simple text extraction.
- Developed a [custom spell-checking post-processing pipeline](https://github.com/jimmmyss/bilingual-spell-checker) for spell and grammar checking using [SymSpell](https://github.com/wolfgarbe/symspell).
- Evaluated [Qwen2.5-VL]([https://github.com/QwenLM/Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL)), [dots.ocr](https://github.com/rednote-hilab/dots.ocr), [NanoNets](https://github.com/NanoNets/docext) VLMs for complex text extraction.

### Discord
- Created a [Discord Community Server](https://discord.com/invite/TY69npdMwM).
- Developed a custom Discord bot that is responsible for announcing and moderating text channels.

### Site
- Developed the official [GlossAPI](https://glossapi.gr) site.

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

The objective was to experimentally evaluate OCR and VLM-based document extraction approaches for Greek text, compare their accuracy, resource requirements, integration complexity and limitations in order to identify which technologies make sense for GlossAPI.

**1. [tesseract](https://github.com/tesseract-ocr/tesseract)**

The initial testing scores showed on average an 70-75% total accuracy score with most notable mistakes occuring in complex text contents such as mathematical formulas, tables, and structured documents. Errors also appeared in simpler text, particularly with unusual polytonic accents or occasional incorrect characters in words.

To address this, a custom post-processing pipeline using Hunspell was developed. This pipeline first strips all accents from the text, then applies Greek spell checking to fix incorrect characters and reapply the correct accents. Additionally, the Tesseract model was trained using in-house data from scrapped PDFs, consisting of paired image files (.tif) and their corresponding ground-truth text files (.gt.txt). With this approach, accuracy improved to over 90% for standard text, though complex content such as mathematical formulas and structured tables remained challenging.

**2. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** 

The initial testing showed a similar average accuracy of 65-70%, with frequent misrecognition of accented characters and polytonic text. While paragraph structure and basic Greek characters were generally preserved, there were instances where entire lines were completely garbled or nonsensical. Compared to Tesseract, PaddleOCR handled noisy or varied fonts slightly better, but it struggled to maintain character-level accuracy and proper accents. Additionally, PaddleOCR is more resource-intensive than Tesseract, as it relies on deep learning models that require more memory and processing power, and it is generally slower on CPU, though GPU inference can speed up batch processing.

To address common recognition errors, a custom post-processing pipeline similar to the one used for Tesseract was developed. This pipeline strips all accents from the text, applies Greek spell checking using Hunspell to correct incorrect characters and reapply the correct accents. Unlike Tesseract, no additional custom training was performed for PaddleOCR at this stage. With this approach, accuracy improved for standard text to 70-75%, though there were still instances of garbled lines. Further training is required to optimize performance for Greek text and effectively eliminate these garbled lines.

**3. [RapidOCR](https://github.com/RapidAI/RapidOCR) & [onnx](https://github.com/onnx/onnx)**

The initial testing demonstrated similar accuracy of 70-75% to PaddleOCR, with slightly better handling of noisy text and significantly fewer garbled or nonsensical line outputs. While it still struggled with accented Greek characters and polytonic text, the overall results were more stable and consistent compared to PaddleOCR alone.
A key advantage of RapidOCR is its compatibility with Docling, which is what's used by GlossAPI for PDF parsing and text extraction. 

Like PaddleOCR, no additional fine-tuning was performed at this stage. However, the ONNX optimization means that RapidOCR can run more efficiently, with potential for GPU acceleration when available.

**4. [Qwen-VL](https://github.com/QwenLM/Qwen-VL), [dots.ocr](https://github.com/rednote-hilab/dots.ocr), [NanoNets](https://github.com/NanoNets/docext)**

The results of these evaluations showed excellent performance on both simple and complex documents, achieving accuracy scores of up to 98% in many instances. Unlike Tesseract and PaddleOCR, these VLMs was able to process mathematical formulas, structured tables, and dense academic content with very high reliability, while also being capable of producing LaTeX representations of equations, making it significantly more effective than the other OCR pipelines.

The main drawback is its computational cost. While Tesseract and PaddleOCR can run efficiently on CPUs or lightweight GPUs (<1 GB VRAM), VLMs requires a modern high-memory GPU (15-40 GB VRAM) and noticeably longer processing times. As GlossAPI is intended for accessibility by the average user rather than being limited to power users with high-end hardware only it was chosen not to be integrated into the GlossAPI pipeline.

**Conclusion**

- After evaluating all approaches, the project selected the [RapidOCR](https://github.com/RapidAI/RapidOCR) approach as the optimal solution for GlossAPI’s OCR requirements. While RapidOCR, built on top of PaddleOCR, was not necessarily a fundamentally more accurate OCR model, it provided a lighter inference path because it run through ONNX Runtime without depending on the full PaddlePaddle framework, and better portability because it could be deployed across different execution environments and hardware backends more easily. Compared to Tesseract, RapidOCR also offered a more scalable path for large-scale document processing, as it can take advantage of GPU-accelerated inference, while Tesseract is primarily CPU-bound during normal OCR usage. Furthermore, because Docling natively supports RapidOCR as one of its built-in OCR engines, integration into the existing GlossAPI pipeline was more straightforward. This combination reduced engineering overhead and made RapidOCR the most practical option for long-term adoption.
- The [custom spell-checking post-processing pipeline](https://github.com/jimmmyss/bilingual-spell-checker) was also evaluated but was not officially integrated into GlossAPI. Although it improved some OCR outputs by correcting common Greek and English recognition errors, tests on academic documents showed that many domain-specific terms, names and technical expressions were missing from the available dictionaries, and as a result, the spell checker sometimes “corrected” words that were already valid, which made it risky for scientific and academic text extraction. For this reason, it was kept as an experimental post-processing tool rather than being adopted as a default part of the GlossAPI pipeline.
- Because of the hardware requirements and the expected user workflows, VLM-based OCR was concluded to be better suited as a separate backend rather than the default extraction path. Users who need VLM OCR are more likely to process large batches of complex PDFs and therefore require dedicated GPU resources, while users relying on the standard Docling pipeline are more likely to need lighter, general-purpose extraction. For that reason, the VLM approach was planned as an optional backend that users could explicitly select when accuracy on complex documents was more important than speed, cost, or hardware accessibility. Since this required a separate architectural integration and was not completed during the GSoC period, the implementation was left as future work. After GSoC concluded, [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) was released in October and was chosen as the more promising candidate for the future VLM-based backend.

## Future Work

The project still has a long way to go before it is ready for mass adoption and capable of handling all types of PDFs. In the near future, I plan to focus on the following improvements:

1. **Expanding Dataset Sources** - Incorporate additional high-quality Greek texts from academic, legal, and public resources to improve coverage.
2. **Pipeline Optimization** - Streamline the entire processing pipeline to maximize efficiency and reduce resource usage.
3. **Selective VLM support for Complex Texts** - Use Vision-Language Models exclusively for complex documents so that standard texts are processed efficiently without heavy resource usage.
4. **Consistent Accuracy** - Achieve and maintain a high OCR accuracy scores of over 90% across diverse document types.

## Final Note 

Thank you [Google Summer of Code 2025](https://summerofcode.withgoogle.com/) and [Open Technologies Alliance - Gfoss](https://gfoss.eu/) for providing me with the opportunity to contribute to GlossAPI. I am deeply grateful for this expirience and to my mentors, Foivos and Nikos, for their guidance and support. The project has given me truly valuable insights into new technologies and methodologies, and I look forward to staying connected with the community and continuing to contribute to GlossAPI.
