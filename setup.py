from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="litterbox-monitor",
    version="1.0.0",
    description="Cat health monitoring agent with CLIP + GPT-4o identification and sensor integration",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    py_modules=["basic_agent", "litterbox_agent"],
    install_requires=[
        "python-dotenv>=1.2.1",
        "tavily-python>=0.7.17",
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
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "bokeh>=3.0.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "bob": ["tavily-python>=0.7.17"],
        # Hardware-facing deps for a real Raspberry Pi deployment. Left empty
        # here because the concrete scale/gas/RFID/camera libraries depend on
        # the specific hardware chosen — add them on-device (e.g. "hx711",
        # "RPi.GPIO", "picamera2"). See docs/PI_SCALE_INTEGRATION.md.
        "pi": [],
        "dev": [
            line
            for line in Path("requirements-dev.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith(("-", "#"))
        ]
        if Path("requirements-dev.txt").exists()
        else [],
    },
    entry_points={
        "console_scripts": [
            "litterbox-agent=litterbox._cli:main",
            "litterbox-bob=litterbox._cli:bob",
            "litterbox-monitor=litterbox.daemon:main",
        ],
    },
    package_data={"litterbox": ["td_config*.json"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Home Automation",
    ],
)
