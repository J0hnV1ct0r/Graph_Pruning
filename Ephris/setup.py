"""Use public image URLs in the package description."""
import re
from pathlib import Path

from setuptools import setup

readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
asset_root = "https://raw.githubusercontent.com/nums-ai/ephris/main/"
readme = re.sub(r"(!\[[^\]]*\]\()(?=docs/)", rf"\g<1>{asset_root}", readme)
readme = readme.replace('src="docs/', f'src="{asset_root}docs/')
setup(long_description=readme, long_description_content_type="text/markdown")
