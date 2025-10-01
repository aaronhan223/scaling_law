#!/usr/bin/env python3
"""
Cache management utility for LibriSpeech quantization analysis.
"""

import argparse
import sys
from data.quantization_cache import get_quantization_cache


def main():
    """Main cache management function."""
    parser = argparse.ArgumentParser(description="Manage LibriSpeech quantization analysis cache")

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./dataset_cache/quantization",
        help="Cache directory path"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status"
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cache files"
    )

    parser.add_argument(
        "--clear-old",
        type=int,
        metavar="DAYS",
        help="Clear cache files older than specified days"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show detailed cache information"
    )

    args = parser.parse_args()

    # Get cache instance
    cache = get_quantization_cache(args.cache_dir)

    if args.status or args.info:
        cache.print_cache_status()

        if args.info:
            info = cache.get_cache_info()
            print("\nDetailed Cache Information:")
            print(f"  Cache directory: {info['cache_dir']}")
            print(f"  Total files: {info['total_files']}")
            print(f"  Total size: {info['total_size_mb']:.2f} MB")
            print(f"  Dataset caches: {info['dataset_caches']}")
            print(f"  Experiment caches: {info['experiment_caches']}")
            print(f"  Metadata files: {info['metadata_files']}")

    elif args.clear:
        print(f"Clearing all cache files from {args.cache_dir}...")
        cache.clear_cache()
        print("Cache cleared successfully!")

    elif args.clear_old is not None:
        print(f"Clearing cache files older than {args.clear_old} days...")
        cache.clear_cache(older_than_days=args.clear_old)
        print("Old cache files cleared successfully!")

    else:
        # Default: show status
        cache.print_cache_status()

        print("\nUsage examples:")
        print("  python manage_quantization_cache.py --status")
        print("  python manage_quantization_cache.py --clear")
        print("  python manage_quantization_cache.py --clear-old 7")
        print("  python manage_quantization_cache.py --info")


if __name__ == "__main__":
    main()