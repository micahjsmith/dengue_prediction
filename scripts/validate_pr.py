#!/usr/bin/env python

def main():
    import dengue_prediction.features.travis as travis
    if is_pr():
        pr_sha = travis.get_pr_sha()
        validate_by_sha(pr_sha)
    else:
        return

if __name__ == '__main__':
    main()
