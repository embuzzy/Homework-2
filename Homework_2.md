Problem 1 - Flights at ABIA
---------------------------

#### I have family that lives in Cleveland, in Houston and in Nashville. I am regularly taking trips to see them. I want to use the Austin airport data to help me plan my trip. First, I'd like to look at which carriers fly directly to these locations, and the number of flights to each location per year:

![](Homework_2_files/figure-markdown_strict/unnamed-chunk-1-1.png)

#### Looks like I don't have that many choices of carriers if I want direct flights to these locations. It also looks like the largest number of flights exist between Austin and Houston. Now I want to use this data to understand when I should plan trips to these 3 cities to avoid delays. First let's take a look at the total delay time for flights going to or returning from each of these cities by month. This will help me understand what time of year I should plan my trips:

![](Homework_2_files/figure-markdown_strict/unnamed-chunk-2-1.png)

#### From this plot, I can see that the best time to travel to Houston is probably sometime in the fall if I want to avoid delays (sept, oct and november) seem to have the least variability and most common occurances around 0 minutes for delay. For travelling to Nashville I can see that January, September and November would be the optimal time for a trip. In these months, it looks like flights tend to leave a bit early and do not have as many delays as other months. Unfortunately, it looks like direct flights to Cleveland ceased in September of 2008, so I might not be able to get a direct flight here anymore, but when they were occuring, it looks like the best months to travel would have been over the summer (specifically July and August) if I wanted the shortest delay.

#### Now let's take a look at the days of the week with the shortest delays for flights TO each city from Austin and flights FROM each city from Austin. This should tell me what days to leave and what days to return in order to avoid delays. First, let's look at delays of flights leaving Austin by destination location/day of the week.

![](Homework_2_files/figure-markdown_strict/unnamed-chunk-3-1.png)

#### From the plot above I can see that the best days to leave Austin for Cleveland are Sunday (day 7) and Monday (day 1), which tend to have flights that leave early. The best day to leave Austin for Houston is Saturday. The best day to leave Austin for Nashville looks to be Saturday or Tuesday - but most of the days seem flights are generally on time. In fact, this seems to be the case for most of the flights departing Austin for these cities. Let's see how flights arriving in Austin from these cities look like. This should give some details on what days of the week I should return to Austin to avoid delays.

![](Homework_2_files/figure-markdown_strict/unnamed-chunk-4-1.png)

#### From the plot above, it seems like the best days to arrive in Austin from Cleveland are either Monday (day 1) or Saturday (day 6). Also from the plot I can see that the day to avoid arriving in to Austin from Cleveland is Sunday, which seems to have a larger number of delays on average. The best days to arrive from Houston look to be Monday, Tuesday or Saturday. The best days to arrive from Nashville look to be Monday or Saturday and similarly to Cleveland, it seems that arriving from Nashville on Sunday tends to have larger delay times.

Problem 2 - Author Attribution
------------------------------

#### First I pulled all of the text out of the list of documents & created a corpus for the training data

    ## Warning in tm_map.SimpleCorpus(my_corpus, content_transformer(tolower)):
    ## transformation drops documents

    ## Warning in tm_map.SimpleCorpus(my_corpus,
    ## content_transformer(removeNumbers)): transformation drops documents

    ## Warning in tm_map.SimpleCorpus(my_corpus,
    ## content_transformer(removePunctuation)): transformation drops documents

    ## Warning in tm_map.SimpleCorpus(my_corpus,
    ## content_transformer(stripWhitespace)): transformation drops documents

    ## Warning in tm_map.SimpleCorpus(my_corpus,
    ## content_transformer(removeWords), : transformation drops documents

Problem 3 - Practice with Association Rule Mining
-------------------------------------------------

#### For this problem, let's first plot the rules output of the apiori algorithm based on lift, support & confidence

![](Homework_2_files/figure-markdown_strict/unnamed-chunk-6-1.png)

#### From this plot, let's pick a range of variables that will give us rules with decent support, confidence & lift. Let's try support&lt;.015, confidence &gt; .35 and lift &gt;2.5

![](Homework_2_files/figure-markdown_strict/unnamed-chunk-7-1.png)

#### From this plot, we can see some clear associations between dairy products - such as butter, sour cream, yogurt & milk. We also see clusterings of fruit and vegetables, meaning purchases of one kind of these items are associated to other kinds of these items. We can also see that there are some items such as whole milk, vegetables & fruit are common in all baskets.
