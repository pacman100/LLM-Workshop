"""
Courtesy: Sayak Paul and Chansung Park.
"""

import os
import subprocess
from multiprocessing import Pool
from github import Github

ORG = "huggingface"
MIRROR_DIRECTORY = "hf_public_repos"
TOK_K = 15


def get_repos(username, access_token=None, include_fork=False):
    """Fetches repositories for a particular GitHub user.

    Courtesy: Chansung Park.
    """
    g = Github(access_token)
    user = g.get_user(username)

    results = []
    for repo in user.get_repos():
        if repo.fork is False:
            results.append((repo.name, repo.stargazers_count))
        else:
            if include_fork is True:
                results.append((repo.name, repo.stargazers_count))
    print(results)
    return results


def sort_repos_by_stars(repos):
    return sorted(repos, key=lambda x: x[1], reverse=True)


def mirror_repository(repository):
    """Locally clones a repository."""
    repository_url = f"https://github.com/{ORG}/{repository}.git"
    repository_path = os.path.join(MIRROR_DIRECTORY, repository)

    # Clone the repository
    subprocess.run(["git", "clone", repository_url, repository_path])


def mirror_repositories():
    # Create the mirror directory if it doesn't exist
    if not os.path.exists(MIRROR_DIRECTORY):
        os.makedirs(MIRROR_DIRECTORY)

    # Get the list of repositories in the organization
    if not os.environ["GH_ACCESS_TOKEN"]:
        raise ValueError("You must set `GH_ACCESS_TOKEN` as an env variable.")
    repositories = get_repos(ORG, os.environ["GH_ACCESS_TOKEN"])
    sorted_repos = sort_repos_by_stars(repositories)
    selected_repos = [x[0] for x in sorted_repos[:TOK_K]]

    print(f"Total repositories found: {len(selected_repos)}.")
    print(selected_repos)
    # Mirror repositories using multiprocessing
    print("Cloning repositories.")
    with Pool() as pool:
        pool.map(mirror_repository, selected_repos)


if __name__ == "__main__":
    mirror_repositories()
