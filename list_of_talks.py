import csv
import numpy as np
'''
This module provides access to the ID's and ratings of valid TED talks
in the dataset. It makes heavy use of the database index (index.csv) sp
please make sure you run the ted_talk_makeindex.py at least once before
using this module.

It offers the following global variables:
@param rating_labels = the labels of the audience ratings in sorted order
@param all_valid_talks = Integer ID's of all the valid TED talks in database
                         Please note that some of the TED videos are not "talks"
                         e.g. dance, music, peotry etc.
@param all_ratings = A dictionary containing the raw rating counts from the
                     audience. Please note that the counts are not normalized.
@param test_set = A set of valid TED talks reserved away as test dataset
'''

# Name of all ratings
rating_labels = sorted(['beautiful','funny','ingenious','ok','fascinating',
  'total_count','persuasive','inspiring','longwinded','informative',
  'jaw-dropping','obnoxious','confusing','courageous','unconvincing'])

# This is a hand curated test set of 150 datapoints. It was randomly sampled
# from the dataset with a constraint that there is at least 2 ratings for each
# type of rating.
test_set = set([ 233,  665,  263,  195,  660,  75, 1315, 1090, 1944,  614, 1565,
        216, 1404, 1633,  516,  483, 1621, 1030,  587, 1326, 2339, 1175,
       1411, 2681, 1034, 1495,  974, 1875, 1985, 1184, 1370, 1008, 1687,
       1575,  869,   41, 1559,  965,  248,   87, 1999, 2704, 2588, 1257,
       2892,  495,   51,   19, 2475, 1643, 1047, 1689, 2417, 1367, 1230,
       2662, 2006, 1614, 1527, 1161, 2395,  850, 2523, 1459, 1052, 1943,
       1328, 1266,  453, 1398, 1859, 1238,  499, 1907, 1581, 2017, 1206,
       1258, 1620, 1836, 2547, 1660,  828,   97, 2608, 2781, 1231, 1195,
       1756,  362, 2055, 2686,  562, 1639, 1503, 2846, 1098, 1553, 1355,
       1801, 2634, 1673, 1285, 2688,  970, 2548,  410,  968,  385, 1067,
        846, 1426, 1555, 1974,  759,  356, 1483, 2570, 2194,  575, 1179,
       1201, 1636,  750, 1463, 1556, 1438,  900, 1853, 1945, 1659, 2566,
       2536,  945,   80,  443, 1532,  675,  633, 2361, 1220,  853, 1546,
       2700,  470, 2391, 1645, 2049, 1517, 1862])

# Process and make the talk id's and talk ratings ready
# Makes the following global variables available:
# all_valid_talks  <-- List of valid talkid's
# all_ratings      <-- List of ratings for the valid talkid's
# test_set_ratings <-- Ratings for the test sets
# all_totalviews   <-- Total view-counts for all the talks
# totalviews_mean  <-- Average of Total view-counts
# totalviews_std   <-- Standard deviation of Total view-counts
reader = csv.DictReader(open('./index.csv','rU'))
all_valid_talks = []
all_ratings = {}
test_set_ratings={}
all_totalviews={}
for arow in reader:
    if arow['Is_a_Talk?']=='Yes':
        atalk = int(arow['Video_ID'])
        all_totalviews[atalk] = int(arow['Totalviews'])
        totalviews_mean = np
        # Skip the talks in the test set. This data is hidden for the final result.
        if test_set and atalk in test_set:
            test_set_ratings[atalk]={ratings:int(arow[ratings]) for ratings in rating_labels}
            continue
        # Skip the talks having partial data. Missing TED_feature_word_boundary in this case
        if atalk in {1379,2744}:
          continue
        all_valid_talks.append(atalk)
        all_ratings[atalk] = {ratings:int(arow[ratings]) for ratings in rating_labels}
totalviews_mean = np.mean(all_totalviews.values())
totalviews_std = np.std(all_totalviews.values())
if test_set:
    test_set=list(test_set)

# A small list of files 
hi_lo_files = {
  'High_View_Talks':[66,1569,848,549,229,96,618,1647,2034,1377,685,
      1246,1344,97,741,206,1821,1815,2405,2399,310,453,652,92,2458,
      2217,1733,1764,1100,70],
  'Low_View_Talks':[524,239,1359,313,318,1263,1341,1452,674,394,
      1294,339,1445,402,500,427,962,268,679,925,1373,403,439,220,
      675,379,345,1466,673,1332]}

allrating_samples = \
[
{'High_Beautiful_Percent':[2300,2635,2304,2556,1416,1603,1604,2511,297,1797,201,1834,2600,2395,1532,2434,2351,2163,1780,500,1435,1171,1442,1283,1140,1634,2117,2336,823,1182],
'Low_Beautiful_Percent':[429,529,641,681,797,1250,1597,408,1350,1638,440,1859,1503,966,1783,1575,521,633,598,1586,1394,470,1656,1673,344,608,2185,1496,925,405]},

{'High_Funny_Percent':[856,1629,203,935,2405,2577,223,87,706,846,2576,114,1776,738,1703,374,148,400,2115,2071,334,149,222,1390,1269,510,86,1371,635,894],
'Low_Funny_Percent':[2235,395,515,1153,2489,1256,783,249,1627,1881,1623,507,736,335,664,1186,1585,2280,1146,358,898,978,869,1003,1361,584,1174,1005,523,1783]},

{'High_Ingenious_Percent':[1559,1901,285,566,623,601,1534,442,664,1941,1665,853,1326,997,760,594,450,245,1616,1295,1470,1598,836,1019,1499,752,1516,1404,2138,1202],
'Low_Ingenious_Percent':[1055,1954,669,1767,2095,1952,1771,1566,1167,586,1541,1997,2569,1003,2018,2165,2171,2024,2488,2137,1473,2110,2681,2472,1654,1972,381,1645,791,963]},

{'High_ok_Percent':[1942,276,1639,552,957,966,2119,473,2061,2197,1159,412,2176,1212,811,1471,2300,282,427,345,1310,1930,503,563,1187,1066,222,680,941,710],
'Low_ok_Percent':[1039,1787,1141,2414,704,233,1115,140,1182,1127,1520,2441,2453,607,2193,2556,2662,33,1898,1685,1110,1687,1666,1530,1672,1683,2594,2472,2627,475]},

{'High_Fascinating_Percent':[551,637,2315,1955,145,1674,77,1853,2048,1758,1139,1628,326,715,1180,1160,2468,724,1075,343,2359,2277,509,1346,571,251,1426,184,1986,2045],
'Low_Fascinating_Percent':[1182,2071,1418,222,1912,1478,1167,2457,1952,1960,963,1136,1096,925,1601,1471,2171,152,1830,1744,203,1176,1012,2596,114,704,2433,149,681,2068]},

{'High_Persuasive_Percent':[152,159,62,192,19,193,47,1480,771,187,1196,116,163,587,803,1473,1642,2472,584,121,1766,1585,1702,1329,1060,1380,1688,797,104,850],
'Low_Persuasuve_Percent':[223,46,267,162,471,87,1993,1604,2159,1834,1508,1942,31,1244,610,2258,534,1969,427,2209,260,1271,1505,562,1442,2230,114,489,345,524]},

{'High_Inspiring_Percent':[1124,2193,1998,1130,2332,1983,498,1183,1355,809,1728,1449,642,1653,1485,2341,1096,1048,2418,2276,1880,2380,1009,1896,735,464,1644,1621,1553,2477],
'Low_Inspiring_Percent':[223,1945,2314,1703,1012,1390,148,1352,345,1250,1597,1107,738,1958,114,203,30,2393,327,602,549,1548,1471,424,1725,1952,2149,529,706,2263]},

{'High_Longwinded_Percent':[520,524,8,231,1003,13,427,402,2581,625,21,673,431,345,1217,1573,692,500,800,44,405,1258,410,174,164,590,845,589,441,1338],
'Low_Longwinded_Percent':[2638,1182,1799,1796,423,1115,1771,220,2362,734,518,2057,292,1744,2385,1982,2617,1144,2553,2052,2528,1777,2569,381,523,239,285,2313,681,408]},

{'High_Informative_Percent':[2132,2272,2317,1148,2593,2149,1725,2177,1503,2548,2668,408,2002,1887,2349,2523,531,2103,10,2656,1630,1563,72,2204,1648,2404,881,2436,2314,2238],
'Low_Informative_Percent':[1814,1285,114,1508,1096,374,400,1456,1797,786,1116,1100,26,1603,46,1110,1068,347,1269,2499,1634,562,610,108,2209,1621,1435,45,223,327]},

{'High_Jaw_Dropping_Percent':[65,178,46,199,1495,140,45,141,82,76,1520,147,327,146,310,786,1376,162,685,796,1103,144,921,469,502,1602,863,206,227,1088],
'Low_Jaw_Dropping_Percent':[474,2016,2162,2287,2039,2669,2394,2200,1930,2642,2171,2300,431,2006,2425,1966,2587,1829,2479,2032,2423,2089,2469,2073,2472,1471,345,313,700,427]},

{'High_Obnoxious_Percent':[589,1012,2185,586,1033,2612,2581,607,388,345,1471,2634,27,271,1413,1526,621,688,1357,495,114,474,1649,1952,1081,2007,701,427,375,359],
'Low_Obnoxious_Percent':[377,1818,1838,2630,1701,2504,2516,414,2385,2126,2052,2363,523,2507,1644,1807,1361,2615,1872,2617,2623,2357,1549,2220,2674,2261,2462,1971,2644,2425]},

{'High_Confusing_Percent':[590,48,351,123,2011,182,31,2185,792,1742,599,371,1484,427,1701,1573,1942,2280,1159,1215,607,396,282,524,197,1922,394,532,190,164],
'Low_Confusing_Percent':[1208,1185,2671,1812,1780,1115,292,821,2562,1672,2320,2157,2407,2586,833,1747,360,1229,1144,1491,1071,379,232,556,1818,2504,2385,1644,1872,2462]},

{'High_Courageous_Percent':[1757,2681,1337,2217,1167,1830,2068,1694,1767,704,2221,2678,2095,299,2231,1654,2171,627,592,1494,1566,918,1596,1950,189,1537,1961,2597,2152,2627],
'Low_Courageous_Percent':[1435,526,2246,1725,1296,362,430,280,670,1834,339,716,715,1057,2356,1490,1111,1701,1142,1628,78,966,1776,1551,705,1823,422,571,2048,300]},

{'High_Unconvincing_Percent':[1952,563,607,2612,411,1719,123,2011,589,358,1745,578,1224,8,862,429,44,811,2614,1942,1394,525,1413,1622,1735,537,1215,1639,532,1338],
'Low_Unconvincing_Percent':[1898,1685,2638,2193,2159,1687,1271,2429,1905,2129,1530,2246,2310,1672,1352,1771,1071,2483,1818,879,2357,6,1019,2385,2664,1872,2430,2052,2390,821]},

{'High_Viewed':[66,1569,848,549,229,96,618,1647,2034,1377,685,1246,1344,97,741,206,1821,1815,2405,2399,310,453,652,92,2458,2217,1733,1764,1100,70],
'Low_Viewed':[524,239,1359,313,318,1263,1341,1452,674,394,1294,339,1445,402,500,427,962,268,679,925,1373,403,439,220,675,379,345,1466,673,1332]}
]


