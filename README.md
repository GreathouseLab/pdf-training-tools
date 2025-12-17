These files are a combination of python learning tools for my sabbatical in AI research science and the basis of the nutrition-oncology Q/A data synthesis. 
- The mupdf_trainer_v2.py file is the primary pipeline for reading, chunking and cleaning pdf files for submitting to the small LLM, located in 'prompts_qa.py', for generation of Q/As
- The duplicate_detector.py file takes the *_qa.jsonl files and uses a sentence transformer to search for identical questions and semantic similars
