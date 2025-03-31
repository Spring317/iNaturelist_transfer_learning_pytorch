import os
import time
from collections import defaultdict
from typing import Optional  # type: ignore

import requests
from bs4 import BeautifulSoup, NavigableString, Tag  # type: ignore
from dataset_builder.core.utility import (  # type: ignore
    SpeciesDict,
    write_data_to_json,
)
from dataset_builder.core.exceptions import FailedOperation
from tqdm import tqdm  # type: ignore


def _extract_species_by_class_web(url: str, verbose: bool = False) -> SpeciesDict:
    """
    Extracts species names grouped by their taxonomic class from a webpage.

    Args:
        url: The URL of the page containing species classification.

    Returns:
        SpeciesDict (Dict[str, List[str]]): Dictionary containing species as keys and their species as values.
    """
    data: SpeciesDict = defaultdict(list)

    try:
        response: requests.Response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {url} cause of: {e}")
        return data

    soup = BeautifulSoup(response.text, "html.parser")

    for section in soup.select("h2.title"):
        class_tag: Optional[Tag] = section.select_one(".othernames .sciname")
        if not class_tag:
            continue

        class_name: str = class_tag.text.strip()
        species_list = section.find_next_sibling("ul", class_="listed_taxa")
        if not isinstance(species_list, Tag):
            continue

        for species in species_list.select("li.clear"):
            scientific_tag: Optional[Tag] = species.select_one(".sciname")

            if scientific_tag:
                scientific_name = scientific_tag.text.strip()
                data[class_name].append(scientific_name)

    if verbose:
        print(f"Extracted {sum(len(v) for v in data.values())} species across {len(data)} classes: {list(data.keys())}")
    return data


def _scrape_species_data(
    total_pages: int, base_url: str, delay: int, verbose: bool = False
) -> SpeciesDict:
    """
    Scrapes species data from multiple pages and aggregates them by taxonomic class.

    Returns:
        SpeciesDict (Dict[str, list[str]]): Dictionary containing species as keys and their species as values.

    Raises:
        requests.RequestsException: The script cannot fetch the page
    """
    all_species: SpeciesDict = defaultdict(list)

    page_range = range(1, total_pages + 1)
    page_iter = page_range if verbose else tqdm(page_range, desc="Scraping pages")

    for page_number in page_iter:
        url = f"{base_url}{page_number}&view=plain"

        try:
            if verbose:
                print(f"Processing Page {page_number}/{total_pages}: ", end="")
                page_species = _extract_species_by_class_web(url, True)
            else:
                page_species = _extract_species_by_class_web(url, False)

            for class_name, species_list in page_species.items():
                all_species[class_name].extend(species_list)

        except requests.RequestException as e:
            print(f"Failed to fetch page {page_number}: {e}")

        time.sleep(delay)

    return all_species


def run_web_crawl(
    base_url: str,
    output_path: str,
    delay: int = 1,
    total_pages: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
):
    """
    Crawls iNaturelist site to scrape species data and saves the results to a JSON file.

    This function scrapes species data from multiple pages of a website, 
    aggregates the data by taxonomic class, and saves it as a JSON file. 
    If the output file already exists and `overwrite` is False, the crawl 
    process will be skipped. Otherwise, the function will fetch the data 
    and store it at the specified location.

    Args:
        base_url: The base URL of the website from which to scrape species data.
        output_path: The file path where the scraped species data will be saved as JSON.
        delay: The delay (in seconds) between requests to avoid overwhelming the server. Defaults to 1.
        total_pages : The number of pages to crawl. Defaults to 1.
        overwrite: Flag to indicate whether to overwrite the existing output file. Defaults to False.
        verbose: Whether to print detailed information during the web crawl process. Defaults to False.

    Raises:
        FailedOperation: If an unexpected error occurs during the web crawling process.
    """
    if os.path.isfile(output_path) and not overwrite:
        print(f"{output_path} already exists, skipping web crawl.")
        return

    try:
        species_data = _scrape_species_data(total_pages, base_url, delay, True if verbose else False)
        write_data_to_json(output_path, "Web crawl results", species_data)
    except KeyboardInterrupt:
        print("Web crawling canceled.")
    except Exception as e:
        raise FailedOperation(f"Unexpected error during web crawl: {e}")
