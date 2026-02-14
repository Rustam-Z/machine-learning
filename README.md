# Machine Learning & Deep Learning & Agentic AI: Roadmap

# Content 
- [Math (Calculus, Linear Algebra, Propability & Statistics)](#math-calculus-linear-algebra-propability--statistics)
- [Python for Data Science](#python-for-data-science)
- [Machine Learning](#machine-learning)
- [Neural Networks and Deep Learning](#neural-networks-and-deep-learning)
- [Agentic AI](#agentic-ai)
- [General](#general-notes)
- [Research](#research)

# Math (Calculus, Linear Algebra, Propability & Statistics)
- [TL;DR Math for Machine Learning](https://www.youtube.com/playlist?list=PLD80i8An1OEGZ2tYimemzwC3xqkU0jKUg)
- [Calculus](https://www.youtube.com/playlist?list=PLmdFyQYShrjd4Qn42rcBeFvF6Qs-b6e-L), *Don't Memorize*
- [Caclulus](https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr), *3Blue1Brown*
- [Linear Algebra](https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), *3Blue1Brown*
- [Statistics & Probability Khan Academy](https://www.khanacademy.org/math/statistics-probability)
- [Calculus, Linear Algebra, and Statistics & Probability](https://www.youtube.com/@JonKrohnLearns/courses)

# Python for Data Science
- [My Python learning roadmap](https://github.com/Rustam-Z/python-programming)
- [NumPy](https://www.w3schools.com/python/numpy/default.asp), [Pandas](https://www.w3schools.com/python/pandas/default.asp), [Matplotlib](https://www.w3schools.com/python/matplotlib_intro.asp) 
- [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)

Books:
- "Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython"
- "Python Data Science Handbook"
      
# Machine Learning
My notes from Stanford course: https://github.com/Rustam-Z/machine-learning-stanford-notes

### What is Machine Learning?
- Machine learning (ML) is field of study that gives computers the ability to learn without being explicitly programmed. Machine Learning is making computers do things that we’ve never made computers do before.
- A computer program is said to learn from *experience E* with respect to some *task T* and some *performance measure P*, if its performance on T, as measured by P, improves with experience E.

### Why use ML?
- Problems for which existing solutions require a lot of hand-tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform better (eg. spam classifier)
- Complex problems for which there is no good solution at all using a traditional approach: the best Machine Learning techniques can find a solution (eg. speech recognition)
- Fluctuating environments: a Machine Learning system can adapt to new data.
- Getting insights about complex problems and large amounts of data (eg. data mining)

### Types of ML Systems:
- Whether or not they are trained with human supervision `supervised, unsupervised, semisupervised, and Reinforcement Learning`
  - **Supervised learning** - training data with labels (expected outputs). 
      - Tasks: classification, regression (univariate / multivariate). 
      - Class / sample / label / feature (predictors: age, brand, ...) / attribute
      - **Algorithms**: k-Nearest Neighbors, Linear Regression, Logistic Regression, Support Vector Machines (SVMs), Decision Trees and Random Forests, Neural networks
  - **Unsupervised learning** - training data is unlabeled.
      - Tasks: clustering, anomaly detection, visualization & dimensionality reduction. 
      - Clustering (find similar visitors): K-Means, DBSCAN, Hierarchical Cluster Analysis (HCA)
      - Anomaly detection & novelty detection (detect unusual things): One-class SVM, Isolation Forest
      - `TIP!` Use dimensionality reduction algo before feeding to supervised learning algorithm.
      - `TIP!` Automatically removing outliers from a dataset before feeding it to another learning algorithm.
  - **Semisupervised learning** - a lot of unlabeled data and a little bit of labeled data. 
      - Example: like in Google photos, it recongnizes same person in many pictures. We need supervised part because we need to seperate similar clusters. (like similar people)
  - **Reinforcement Learning** - *agent* can observe environment, and perform some actions, and get *rewards* and *penalties*. Then it must teach itself the best strategy (*policy*) to get max reward. A policy defines what action the agent should choose when it is in a given situation.

Courses:
- *[Machine Learning Course, Andrew Ng, coursera.org](https://www.coursera.org/specializations/machine-learning-introduction)
  - [Machine Learning Stanford Cheatsheet](https://stanford.edu/~shervine/teaching/cs-229/)
- [Google's ML Crash Course](https://developers.google.com/machine-learning/crash-course) - Just for fast recapping
- [Learn Intro to Machine Learning | Kaggle](https://www.kaggle.com/learn/intro-to-machine-learning)
- *[Top Machine Learning Courses](https://www.learndatasci.com/best-machine-learning-courses)
- *[How to Learn Machine Learning](https://elitedatascience.com/learn-machine-learning)
- [Stanford CS221: Artificial Intelligence](https://www.youtube.com/playlist?list=PLoROMvodv4rO1NB9TD4iUZ3qghGEGtqNX)
- [Stanford CS229: Machine Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
- [Amazon Machine Learning Guide](https://docs.aws.amazon.com/machine-learning/latest/dg/machinelearning-dg.pdf)
- [Krish Naik's complete ML course](https://www.youtube.com/playlist?list=PLZoTAELRMXVPBTrWtJkn3wWQxZkmTXGwe)

Books:
- *"Machine Learning For Absolute Beginners" Oliver Theobald
- *"Machine Learning for Humans" - all in one, very short explanation of ML
- *"The Hundred-Page Machine Learning Book" Andriy Burkov
- *“Deep Learning with Python” - first chapters 
- ***["Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow"](https://github.com/ageron/handson-ml2) - first chapters
- "Machine Learning Engineering" Andriy Burkov
- "The Elements Of Statistical Learning: Data Mining, Inference and Prediction"
- "AI and Machine Learning for Coders" - Laurence Moroney, deeplearning.ai TensorFlow Developer specialization instructor
- ["Python Machine Learning"](https://github.com/rasbt/python-machine-learning-book)
- *["Machine Learning Yearning"](https://github.com/Rustam-Z/deep-learning-notes/blob/main/andrew-ng-ml-book.pdf) Andrew Ng - After finishing this book, you will have a deep understanding of how to set technical direction for a machine learning project.

Practice:
- *[Applied Machine Learning](https://machinelearningmastery.com/start-here)
- [The Mechanics of Machine Learning](https://mlbook.explained.ai/)
- [Practical Machine Learning with Python](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)
- **Scikit-Learn**
    - https://inria.github.io/scikit-learn-mooc/
    - https://scikit-learn.org/stable/tutorial/index.html
          
# Neural Networks and Deep Learning 
My notes from Deep Learning course by Andrew Ng: https://github.com/Rustam-Z/deep-learning-notes

Courses:
- *[Deep Learning Specialization, Andrew Ng, coursera.org](https://www.coursera.org/specializations/deep-learning)
- www.deeplearning.ai
- [CS230: Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb) - A class of DL at Stanford by Andrew Ng
- [MIT Deep Learning](http://introtodeeplearning.com/)
- [Krish Naik's complete DL course](https://www.youtube.com/playlist?list=PLZoTAELRMXVPGU70ZGsckrMdr0FteeRUi) - In case you get stuck and don't understand the concepts try to find the easy explained video in this playlist
- *[Andrej Karpathy YouTube videos](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- Stanford and MIT free courses
  
Books:
- *["Grokking Deep Learning"](https://t.me/progbook2/216)
- *["Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow"](https://github.com/ageron/handson-ml2)
- *["Deep Learning with Python"](http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf)
- "Deep Learning for Coders with fastai and PyTorch"
- `advanced` "Deep learning", MIT press, "Written by three experts in the field, Deep Learning is the only comprehensive book on the subject."

Extra:
- [Cheat Sheets for AI, Neural Networks, Machine Learning, Deep Learning & Big Data](https://becominghuman.ai/cheat-sheets-for-ai-neural-networks-machine-learning-deep-learning-big-data-678c51b4b463) `numpy`, `pandas`, `sklearn`, `ml`, `dl`

# Agentic AI

At the heart of agentic AI are autonomous agents that can:
- Perceive: Interpret input from the environment.
- Plan: Generate strategies to achieve goals.
- Act: Execute actions and interact with the world.
- Learn: Improve decisions based on feedback.

Start with [Intro to LLM by Andrej Karpathy](https://www.youtube.com/playlist?list=PLAqhIrjkxbuW9U8-vZ_s_cjKPT_FqRStI) course.

To read:
- [The Roadmap for Mastering Agentic AI in 2026](https://machinelearningmastery.com/the-roadmap-for-mastering-agentic-ai-in-2026/?ref=dailydev#)
- https://modelcontextprotocol.io
- https://code.claude.com/docs
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- https://agentskills.io/integrate-skills
- https://geminicli.com/docs
- https://geminicli.com/docs/cli/skills
- https://google.github.io/adk-docs

To watch:
- [Intro to LLM by Andrej Karpathy](https://www.youtube.com/playlist?list=PLAqhIrjkxbuW9U8-vZ_s_cjKPT_FqRStI)
- [Stanford LLM course](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy)
- https://www.youtube.com/@aiexplained-official/videos
- https://www.youtube.com/@ShawhinTalebi/videos
- https://www.youtube.com/@umarjamilai/videos

# General notes
- https://roadmap.sh/ai-engineer
- https://roadmap.sh/ai-agents
- https://roadmap.sh/prompt-engineering
- https://huggingface.co/learn
- https://www.kaggle.com/learn
- https://machinelearningmastery.com/start-here
- https://towardsdatascience.com
- https://www.deeplearning.ai
- https://www.fast.ai

# Research
- [Anthropic](https://www.anthropic.com/research)
- [OpenAI](https://openai.com/blog/tags/research/)
- [Google DeepMind](https://deepmind.com/research)
- [Google AI](https://ai.google/research/)
- [Google AI Blog](https://ai.googleblog.com/)
- [Stanford AI Lab](https://ai.stanford.edu)
- [MIT AI Lab](https://www.csail.mit.edu)
- [Microsoft Research](https://www.microsoft.com/en-us/research/research-area/artificial-intelligence)
- [IBM Research](https://www.research.ibm.com/artificial-intelligence/#publications)
