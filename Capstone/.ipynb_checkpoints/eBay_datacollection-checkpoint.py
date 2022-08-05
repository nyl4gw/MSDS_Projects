## NMICP Data Engineering Pipeline

import pandas as pd
import numpy as np
import os
import requests
import dotenv
import json
import base64
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
import xmltodict
from collections import OrderedDict
import hashlib
import traceback

import sqlite3
import os
import time 

#import categories list, make sure file is in same directory
from CategoryList_Input import categories_of_interest

#set up API
#Load secret file that contains API keys
dotenv.load_dotenv('keys.env')

#Load in secret keys 
AppID = os.getenv('AppID')
DevID = os.getenv('DevID')
CertID = os.getenv('CertID')

#Combine AppID and CertID to create encoded authorization token
s = AppID + ':' +CertID
encoded = base64.b64encode(s.encode('UTF-8'))

#URL that verifies API user
url = 'https://api.ebay.com/identity/v1/oauth2/token'

#Parameters passed to the API connection
headers = {'Authorization': 'Basic ' + str(encoded.decode("utf-8")),
          'Content-Type': 'application/x-www-form-urlencoded'}

#Connect to API
params = {'grant_type':'client_credentials',
         'scope': 'https://api.ebay.com/oauth/api_scope'}

r = requests.post(url, headers=headers, params=params)
r

#Retrieve OAuth token for connecting to the Shopping API later
OAuth = json.loads(r.text)['access_token']
oneday = pd.to_datetime(date.today() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S.000Z')

#--- BEGIN TRY BLOCK

try:

    #Functions that collect data
    def geteBay(categoryid, starttime):

        #Function to connect to eBay's Finding API and collect item listings for the given category
        url = 'https://svcs.ebay.com/services/search/FindingService/v1' #URL that links to the Finding API
        headers = {'X-EBAY-SOA-SECURITY-APPNAME': AppID,
                   'X-EBAY-SOA-OPERATION-NAME': 'findItemsByCategory'} #AppID is a global variable in the main script
        
        #Initialize empty dataframe
        categorydf = pd.DataFrame()

        #Parse through desired numbers of pages in the Finding API (Given Call limits we have limited this to 2 pages)
        for i in range(1,2):
            params = {'categoryId': categoryid,
                     'RESPONSE-DATA-FORMAT':'JSON',
                     'paginationInput.entriesPerPage':100, #100 entries per page, This is the maximum number of entries we can get for any one category
                     'paginationInput.pageNumber':i,
                     'findItemsByCategoryRequest.sortOrder':'StartTimeNewest',
                      'itemFilter(0).name': 'StartTimeFrom',
                      'itemFilter(0).value': starttime}
            
            #Make GET request call to Finding API
            r = requests.get(url, headers=headers, params=params)
            
            #Continue to next page if nothing was collected
            if 'item' not in pd.json_normalize(json.loads(r.text)['findItemsByCategoryResponse'][0]['searchResult'][0]):
                continue 
                
            #Add item listings to dataframe from current page
            categorydf = categorydf.append(pd.json_normalize(json.loads(r.text)['findItemsByCategoryResponse'][0]['searchResult'][0]['item']))

        #Reset the row index in the item dataframe
        categorydf = categorydf.reset_index()

        #Cleaning Dataframe
        #Iterate through each listing of the dataframe and clean corresponding column values
        #item ID
        itemlist = []
        for i in range(0, len(categorydf)): #loop over range of categorydf
            if 'itemId' in categorydf: #if current row has an associated itemId...
                item = categorydf.itemId[i][0] #extract itemId...
            else:
                item = 'nan' #else label as nan
            itemlist.append(item)

        #title
        titlelist = []
        for i in range(0, len(categorydf)):
            if 'title' in categorydf:
                title = categorydf.title[i][0]
            else: 
                title = 'nan'
            titlelist.append(title)

        #viewitemurl,  this is NOT the image URL, it is the link to the actual listing
        urllist = []
        for i in range(0, len(categorydf)):
            if 'viewItemURL' in categorydf:
                url = categorydf.viewItemURL[i][0]
            else: 
                url = 'nan'
            urllist.append(url)

        #postalcode
        postallist = []
        for i in range(0, len(categorydf)):
            if 'postalCode' in categorydf:
                 code = categorydf.postalCode[i]
            else:
                 code = 'nan'
            postallist.append(code)

        postal = []
        for i in postallist:
            if type(i) == list:
                postal.append(i[0])
            else:
                postal.append("nan")


        #country
        countrylist = []
        for i in range(0, len(categorydf)):
            if 'country' in categorydf:
                a = categorydf['country'][i][0]
            #print(a)
            else:
                a = 'nan'
            countrylist.append(a)

        #priceselling
        pricelist = []
        for i in range(0, len(categorydf)):
            if 'sellingStatus' in categorydf:
                price = categorydf.sellingStatus[i][0]['convertedCurrentPrice'][0]['__value__']
            else:
                price = 'nan'
            pricelist.append(price)

        #condition
        conditionlist = []
        for i in range(0, len(categorydf)):
            if 'condition' in categorydf:
                a = categorydf['condition'][i]
            else: 
                a = 'nan'
            conditionlist.append(a)

        #listingtime
        listingtime = []
        for i in range(0, len(categorydf)):
            if 'listingInfo' in categorydf:
                a = categorydf['listingInfo'][i][0]['startTime'][0]
            else:
                a = 'nan'
            listingtime.append(a)

        #create final version of dataframe from cleaned item listings
        categorydf_clean = pd.DataFrame({'Item_ID': itemlist,
                                         'Product_Title':titlelist,
                                         'URL_image':urllist,
                                         'Country':countrylist,
                                         'Price_USD':pricelist,
                                         'Postal_Code': postal,
                                         'Item_Condition': conditionlist,
                                         'Listing_Time':listingtime})

        return categorydf_clean

    #function to change data type to a dictionary
    def OrderedDict_to_dict(arg):
        if isinstance(arg, (tuple, list)): 
            return [OrderedDict_to_dict(item) for item in arg]

        if isinstance(arg, OrderedDict): 
            arg = dict(arg)

        if isinstance(arg, dict): 
            for key, value in arg.items():
                arg[key] = OrderedDict_to_dict(value)

        return arg

    #Assign categories of interest to categories_list object
    categories_list = categories_of_interest

    os.chdir('/gpfs/gpfs0/project/nmicp')
    ebay_db = sqlite3.connect("ebaydata.db")

    # LOOP ACROSS CATEGORIES OF INTEREST
    for cat in categories_list:
        #Initialize item listing dataframe using our geteBay function
        finding_df = geteBay(cat, oneday)
        #Extract item IDs from dataframe to use in Shopping API call
        itemlist = []
        for i in range(0, len(finding_df['Item_ID'])):
            item = finding_df['Item_ID'][i]
            itemlist.append(item)

        #Section itemlist into sets of 20 item IDs to adhere to API call limit
        new_list = [itemlist[i:i + 20] for i in range(0, len(itemlist), 20)]

        root = 'https://open.api.ebay.com'
        endpoint = '/shopping'
        headers = {'X-EBAY-API-IAF-TOKEN': 'Bearer ' + OAuth,
                  'Content-Type': 'application/x-www-form-urlencoded',
                  'Version': '1199'}
        
        #Initialize empty dataframe for Shopping API results
        getmultipledf = pd.DataFrame()
        
        #Parse through sets of ItemIDs and add item information to dataframe
        for eachlist in new_list:
            string_listingids = ','.join(eachlist)
            params = {'callname':'GetMultipleItems',
                    'ItemID': string_listingids,
                    'IncludeSelector':'Variations,Details,ItemSpecifics'}
           
            #Make GET request call to Shopping API
            r = requests.get(root+endpoint, headers=headers, params=params)
            xml_format = xmltodict.parse(r.text)
            
            #Append item information to dataframe
            getmultipledf = getmultipledf.append(pd.json_normalize(OrderedDict_to_dict(xml_format)))

        #Reset row index of dataframe   
        getmultipledf = getmultipledf.reset_index()

        #If no new items were found for current category, continue to the next one
        if 'GetMultipleItemsResponse.Item' not in getmultipledf:
            continue 

        #Initialize empty lists for item listing features
        item_specs = []
        item_idlist = []
        seller_id = []
        item_sku = []
        image_url = []
        category_id = []

        #Check that the object is a list, otherwise cast it into one
        convert_list = getmultipledf['GetMultipleItemsResponse.Item']
        convert_list = [convert_list] if isinstance(convert_list, float) else convert_list

        #Iterate through item pages
        for j in range(0, len(convert_list)): #loop over length of convert_list
            
            #skip empty elements
            if type(convert_list[j]) == float:
                continue
            
            for i in range(0, len(convert_list[j])):
                a = getmultipledf['GetMultipleItemsResponse.Item'][j][i]['PrimaryCategoryID'] 
                category_id.append(a)

            #item specifics
            for i in range(0, len(convert_list[j])):
                a = getmultipledf['GetMultipleItemsResponse.Item'][j][i]['ItemSpecifics']['NameValueList']
                item_specs.append(a)

            #item ID
            for i in range(0, len(convert_list[j])):
                a = getmultipledf['GetMultipleItemsResponse.Item'][j][i]['ItemID']
                item_idlist.append(a)

            # seller ID (encrypted)
            for i in range(0, len(convert_list[j])):
                a = getmultipledf['GetMultipleItemsResponse.Item'][j][i]['Seller']['UserID']
                m = hashlib.sha256(a.encode('utf8'));
                seller_id.append(m.hexdigest())

            #SKU
            for i in range(0, len(convert_list[j])):
                a  = getmultipledf['GetMultipleItemsResponse.Item'][j][i].get('SKU')
                item_sku.append(a)

            #image URL
            all_picture_urls = []
            for i in range(0, len(convert_list[j])):
                if isinstance(getmultipledf['GetMultipleItemsResponse.Item'][j][i]['PictureURL'], list):
                    all_picture_urls.append(getmultipledf['GetMultipleItemsResponse.Item'][j][i]['PictureURL'])
                else:
                    converted_to_list = [getmultipledf['GetMultipleItemsResponse.Item'][j][i]['PictureURL']]
                    all_picture_urls.append(converted_to_list)

            for i in range(0, len(all_picture_urls)):
                a = all_picture_urls[i][0]
                image_url.append(a)

        #Create dataframe from Shopping API data
        shopping_df = pd.DataFrame({'itemspeclist': item_specs,
                               'itemid': item_idlist,
                               'sellerid':seller_id,
                              'sku':item_sku,
                              'image_url':image_url,
                                   'categoryid': category_id})

        #Combine Finding and Shopping API data in one dataframe
        item_specs = pd.DataFrame({'ItemID':finding_df['Item_ID'],
                                  'Product_Title':finding_df['Product_Title'],
                                  'CategoryID':shopping_df['categoryid'],
                                  'Price':finding_df['Price_USD'],
                                  'Item_Condition': finding_df['Item_Condition'].astype('str'),
                                  'Listing_Time':finding_df['Listing_Time'].astype('str'),
                                  'Item_Specifics':shopping_df['itemspeclist'].astype('str'),
                                  'Seller_ID':shopping_df['sellerid'],
                                  'Country':finding_df['Country'],
                                  'Zip_Code':finding_df['Postal_Code'],
                                  'Image_URL':shopping_df['image_url'],
                                  'SKU':shopping_df['sku']})

        #Append rows from combined dataframe to SQL database
        item_specs.to_sql("item_information", ebay_db, index=False, chunksize=1000, if_exists="append")
      

    #Commit changes to database and close connection
    ebay_db.commit()
    ebay_db.close()
    
#--- END TRY BLOCK

#--- BEGIN EXCEPT BLOCK 

#If there is a ConnectionError at any point, skip TRY block and execute next two lines
except ConnectionError:
    print("A connection error occurred!")
    
#If there is an error besides a ConnectionError at any point, skip TRY block and execute following code   
except Exception as e:
    categoryerror = item_specs.CategoryID.iloc[-1] #extracts most recent CategoryID
    
    with open('Ebay_Script_Log.txt', 'a+') as f: #w will overwrite existing content in the log.txt file
        f.write(str(e)) #Add error name
        f.write(traceback.format_exc()) #Add the error's entire traceback message
        #add line that extracts cat iteration
        f.close()
    print("Something other than a ConnectionError happened")
    print("Error occured at category " + str(categoryerror))
    #Troubleshooting:
    #Check result.out file first, see if something happened 
    #If something happened, check Ebay_Script_Log.txt file to see traceback message 
   
    #Commit changes to database and close connection (this will ensure all previously extracted listings are properly saved)

    ebay_db.commit()
    ebay_db.close()
    
#--- END EXCEPT BLOCK 
