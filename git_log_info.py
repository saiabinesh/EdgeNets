import subprocess

def run_git_command(command):
    """Helper function to run a Git command and return its output."""
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.stderr}")
        return None

def get_git_log_info():
    # Get the first commit in the repository
    first_commit = run_git_command(["git", "rev-list", "--max-parents=0", "HEAD"])
    
    # Get the current commit ID
    current_commit = run_git_command(["git", "rev-parse", "HEAD"])
    
    # Get the difference between the first and current commit
    if first_commit and current_commit:
        diff = run_git_command(["git", "diff", first_commit, current_commit])
        diff_stat = run_git_command(["git", "diff", "--stat", first_commit, current_commit])
    else:
        diff = None
        diff_stat = None

    # Output the results
    print(f"First Commit: {first_commit}")
    print(f"Current Commit: {current_commit}")
    if diff_stat:
        print("\nDifference Summary (Stat):\n", diff_stat)
    if diff:
        print("\nFull Difference:\n", diff)

if __name__ == "__main__":
    get_git_log_info()
