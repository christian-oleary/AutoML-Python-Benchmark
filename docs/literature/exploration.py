"""Code for literature exploration.

- Find papers related to Python AutoML libraries.
- Scan papers for mentions of Python AutoML libraries.
- Compile a matrix of papers and libraries, indicating which papers mention which libraries.
"""

from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
from pyalex import OpenAlexResponseList, Works  # pip install pyalex

from ml import display_names, package_names

# pyalex.config.api_key = "<YOUR_API_KEY>"

LIBRARIES = set(display_names.values()) | set(package_names.keys())
TOPICS = set(
    [
        'anomaly detection',
        'forecast',
        'time series',
        'time series anomaly detection',
        'time series classification',
        'time series forecasting',
    ]
)


def search_papers(
    libraries: set[str],
    max_results: int = 200,
    min_year: int = 2015,
    output_dir: str | Path = 'results/literature/papers_by_library',
    topics: set[str] | None = None,
):
    """Search for papers mentioning the Python AutoML libraries.

    If there are too many results, the papers are filtered by keywords and then
    sorted by year to find the most recent ones. The results are stored in a
    matrix indicating which papers mention which libraries.

    :param set[str] libraries: Set of library names to search for
    :param int max_results: Maximum number of papers to retrieve per library, defaults to 200
    :param int min_year: Minimum publication year for papers, defaults to 2015
    :param str | Path output_dir: Directory to save results
    :param set[str] | None topics: Optional set of topics to filter papers by
    """
    for library in libraries:
        # Check if results already exist for this library to avoid hitting API limits
        results_path = Path(output_dir) / f'{library.replace(" ", "_")}_papers.json'
        if results_path.exists():
            with results_path.open('r', encoding='utf-8') as f:
                results = json.load(f)
            continue

        # Search for papers mentioning the library, filtering by publication year
        query = Works().search(library).filter(publication_year=f'>{min_year}')

        # If there are too many results, filter by topics
        if topics:
            topic_query = ' OR '.join(topics)
            query = query.search(topic_query)
        results: OpenAlexResponseList = query.get(per_page=max_results)

        # Save results to file to prevent hitting API limits in future runs
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)


def compile_results(
    output_dir: str | Path = 'results/literature',
    papers_dir: str | Path = 'results/literature/papers_by_library',
) -> dict:
    """Compile results into a matrix indicating which papers mention which libraries.

    :param str | Path output_dir: Directory to save compiled results
    :param str | Path papers_dir: Directory where individual paper results are stored
    :return dict: Dictionary containing the compiled results
    """
    results_all_libraries = []
    for library in LIBRARIES:
        # Load results for this library
        results_path = Path(papers_dir) / f'{library.replace(" ", "_")}_papers.json'
        if results_path.exists():
            with results_path.open('r', encoding='utf-8') as f:
                papers = json.load(f)
        else:
            print(f'No results found for library "{library}" at {results_path}')
            continue

        paper_results = []
        for paper in papers:
            data = {'id': paper['id'], 'doi': paper['doi']}
            # Top-level fields
            for key in [
                'title',
                'publication_year',
                'publication_date',
                'relevance_score',
                'type',
                'FWCI',  # Field-Weighted Citation Impact
                'cited_by_count',
                'is_retracted',
                'referenced_works_count',
            ]:
                value = paper.get(key, '')
                if value is None:
                    value = ''
                if key == 'title':
                    value = value.replace('\n', ' ').replace(',', ' ').strip()
                data[key] = value

            # Nested fields
            fields = {
                'citation_normalized_percentile': ['value'],
                'has_content': ['pdf'],
                'primary_location': ['raw_source_name', 'landing_page_url', 'raw_type'],
                'content_urls': ['pdf'],
            }
            for field, subfields in fields.items():
                if field in paper and isinstance(paper[field], dict):
                    for subfield in subfields:
                        data[f'{field}__{subfield}'] = paper[field].get(subfield, '')

            # Authors
            if 'authorships' in paper:
                data['authors'] = ''
                for authorship in paper['authorships']:
                    data['authors'] += f'{authorship["author"]["display_name"]}; '
            paper_results.append(data)

        # Create a DataFrame for this library's results and sort by relevance score and publication year
        df_paper_results = pd.DataFrame(paper_results)
        df_paper_results['library'] = library
        sort_cols = [
            c for c in ['relevance_score', 'publication_year'] if c in df_paper_results.columns
        ]
        if len(sort_cols) > 0:
            df_paper_results = df_paper_results.sort_values(
                by=sort_cols, ascending=[False] * len(sort_cols)
            )

        # Save results for this library to CSV
        csv_path = Path(papers_dir) / f'{library.replace(" ", "_")}_papers.csv'
        df_paper_results.to_csv(csv_path, index=False)
        # Add library results to the overall list
        results_all_libraries.extend(paper_results)

    # Create a DataFrame, drop duplicates, and sort by relevance score and publication year
    df = pd.DataFrame(results_all_libraries)
    df = df.drop_duplicates()
    sort_cols = [c for c in ['relevance_score', 'publication_year'] if c in df.columns]
    if len(sort_cols) > 0:
        df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

    # Save the compiled results to a CSV file
    results_path = Path(output_dir) / 'compiled_results.csv'
    df.to_csv(results_path, index=False)

    # Check the format of the compiled results
    df = pd.read_csv(results_path)
    print(df.head(), '\n')
    print(df.columns, '\n')
    print(df.info(), '\n')
    return df


# def analyze_results(compiled_results_path: str | Path = 'results/literature/compiled_results.csv') -> None:
#     """Perform analysis on the compiled results.

#     This function can be used to analyze the compiled results, such as counting
#     how many papers mention each library, identifying trends over time, etc.

#     :param str | Path compiled_results_path: Path to the compiled results CSV file
#     """
#     df = pd.read_csv(compiled_results_path)

#     # Count how many papers mention each library
#     library_counts = df['library'].value_counts()
#     print('Number of papers mentioning each library:')
#     print(library_counts)

#     # Count how many papers mention each library by year
#     library_year_counts = df.groupby(['publication_year', 'library']).size().unstack(fill_value=0)
#     print('Number of papers mentioning each library by year:')
#     print(library_year_counts)

#     # For each paper and each library, check if the paper mentions the library
#     for _, row in df.iterrows():
#         paper_id = row['id']
#         for lib in LIBRARIES:
#             if row[f'mentions_{lib}'] == 1:
#                 print(f'Paper {paper_id} mentions library "{lib}"')


if __name__ == '__main__':
    OUTPUT_DIR = 'results/literature/'
    PAPERS_DIR = 'results/literature/papers_by_library'
    search_papers(LIBRARIES, min_year=2015, output_dir=PAPERS_DIR, topics=TOPICS)
    # search_papers(LIBRARIES, min_year=2015, output_dir=PAPERS_DIR, topics=LIBRARIES)
    # search_papers(LIBRARIES, min_year=2015, output_dir=PAPERS_DIR, topics=None)
    compile_results(output_dir=OUTPUT_DIR, papers_dir=PAPERS_DIR)
