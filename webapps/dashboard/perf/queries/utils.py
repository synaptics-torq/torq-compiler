from django.db import connection


def get_histogram(sql_query, field, fields, num_bins, min_value, max_value):
    """
    Compute a histogram of the given field for the given queryset using PostgreSQL's width_bucket function.
    """
    
    # Build histogram from the queryset SQL as a derived table:
    # SELECT width_bucket(subq.<field>, ...) ... FROM (<qs SQL>) AS subq
    
    histogram_sql = f"""
        SELECT
            width_bucket(subq.\"{field}\"::double precision, %(max_value)s, %(min_value)s, %(num_bins)s) AS bucket,
            COUNT(*) AS count
        FROM ({sql_query}) AS subq
        GROUP BY bucket
        ORDER BY bucket
    """

    fields['num_bins'] = num_bins
    fields['min_value'] = min_value
    fields['max_value'] = max_value

    # we use inverted bins to make sure 0 ends up in the negative bin (which is the "good bin")
    with connection.cursor() as cursor:
        cursor.execute(histogram_sql, fields)
        rows = cursor.fetchall()

    count_per_bucket = {row[0]: row[1] for row in rows}

    entries = []

    for bucket in range(num_bins + 2):

        # we want to display the buckets in increasing order
        # so we need to invert the bucket number returned by
        # width_bucket
        inverted_bucket = num_bins + 1 - bucket

        if bucket == 0:
            # underflow bucket
            entries.append({
                "lower_bound": None,
                "upper_bound": min_value,
                "bucket": bucket,
                "count": count_per_bucket.get(inverted_bucket, 0),
                "label": f"≤ {min_value}%"
            })
        elif bucket == num_bins + 1:
            # overflow bucket
            entries.append({
                "lower_bound": max_value,
                "upper_bound": None,
                "bucket": bucket,
                "count": count_per_bucket.get(inverted_bucket, 0),
                "label": f">{max_value}%"
            })
        else:
            # regular bucket
            lower_bound = min_value + (max_value - min_value) * (bucket - 1) / num_bins
            upper_bound = min_value + (max_value - min_value) * bucket / num_bins
            entries.append({
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "bucket": bucket,
                "count": count_per_bucket.get(inverted_bucket, 0),
                "label": f"( {lower_bound:.1f}% , {upper_bound:.1f}% ]"
            })

    return entries


def get_average(sql_query, field, fields):
    """
    Compute the average of the given field for the given queryset.
    """

    average_query = f"""
        SELECT AVG(subq.\"{field}\"::double precision) AS average
        FROM ({sql_query}) AS subq
    """

    with connection.cursor() as cursor:
        cursor.execute(average_query, fields)
        row = cursor.fetchone()

    return row[0]

