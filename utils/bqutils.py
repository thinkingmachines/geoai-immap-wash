# # login
# from google.oauth2 import service_account
# # create service account in IAM-Admin
# # source: https://cloud.google.com/bigquery/docs/authentication/service-account-file
# !gcloud iam service-accounts keys create name123456credentials.json --iam-account cholo-bq@immap-colombia-270609.iam.gserviceaccount.com
# credentials = service_account.Credentials.from_service_account_file(
#     'name123456credentials.json',
#     scopes=["https://www.googleapis.com/auth/cloud-platform"],
# )

# client = bigquery.Client(
#     credentials=credentials,
#     project=credentials.project_id,
# )

# main function
from google.cloud import bigquery
project = 'immap-colombia-270609'
dataset = 'wash_prep'
client = bigquery.Client()

def bq(table_name, query, dataset = dataset, create_view = True, dry_run = False, client = client):
    """Read the docs!! https://googleapis.dev/python/bigquery/latest/usage/tables.html"""
    # configure bq
    # client = bigquery.Client(project)
    job_config = bigquery.QueryJobConfig()
    if create_view == True:
        # delete view if exists
        table_id = project + '.' + dataset + '.' + table_name + '_view'
        client.delete_table(table_id, not_found_ok=True)
        table_ref = client.dataset(dataset).table(table_name + '_view')
        table = bigquery.Table(table_ref)
        table.view_query = query
        table.view_use_legacy_sql = False
        client.create_table(table)
        print("Successfully created view at {}".format(table_name + '_view'))
    if dry_run == True:
        job_config.dry_run = True
        job_config.use_query_cache = False
        query_job = client.query(query, location='US', job_config=job_config)
        # A dry run query completes immediately.
        assert query_job.state == "DONE"
        assert query_job.dry_run
    else:
        table_ref = client.dataset(dataset).table(table_name)
        job_config.destination = table_ref
        # delete table if exists
        table_id = project + '.' + dataset + '.' + table_name
        client.delete_table(table_id, not_found_ok=True)
        # start the query, passing in the extra configuration
        query_job = client.query(query, location='US', job_config=job_config)
        # Waits for the query to finish
        rows = query_job.result()
        print('Query results loaded to table {}'.format(table_ref.path))
        table = client.get_table(client.dataset(dataset).table(table_name))
        print('Number of rows: {}'.format(table.num_rows))
    print("This query will process {} MB.".format(query_job.total_bytes_processed/1024/1024))

# bq('', """

# """)

def run_sql(filename, dataset = dataset, replace = None):
    '''
    Runs a sql file, affecting BQ dataset, replacing parts of the script
    
    Args
        replace (dict): in the format {'str_to_replace': 'replacement'}
    '''
    with open(filename) as file:
        script = file.read()
    if replace is not None:
        for k, v in replace.items():
            script = script.replace(k, v)
    for snippet in script.split(';'):
        # check first if it's a valid SQL query.
        if re.search('(select)|(from)', snippet, re.IGNORECASE):
            # that should already be the query
            query = snippet.strip()
            # get table name
            first_line = query.split('\n')[0]
            table_name = first_line[first_line.find(': ')+2:]
            # execute query
            bq(
                table_name,
                query,
                dataset
            )
            print()