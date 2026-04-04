from setuptools import setup, find_packages

setup(
    name="litterbox-monitor",
    version="1.0.0",
    description="Cat health monitoring agent with CLIP + GPT-4o identification and sensor integration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "python-dotenv>=1.2.1",
        "langchain>=1.2.0",
        "langchain-classic>=1.0.1",
        "langchain-core>=1.2.6",
        "langchain-openai>=1.1.6",
        "langgraph>=1.0.5",
        "langgraph-checkpoint>=3.0.1",
        "langgraph-checkpoint-sqlite>=3.0.3",
        "chromadb>=0.6.0",
        "sentence-transformers>=3.0.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "bob": ["tavily-python>=0.7.17"],
        "dev": open("requirements-dev.txt").read().splitlines()
        if __import__("pathlib").Path("requirements-dev.txt").exists()
        else [],
    },
    entry_points={
        "console_scripts": [
            "litterbox-agent=litterbox._cli:main",
            "litterbox-bob=litterbox._cli:bob",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Home Automation",
    ],
)
