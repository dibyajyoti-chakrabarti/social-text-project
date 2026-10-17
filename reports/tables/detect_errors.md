# Detection errors on gold test (n=300)

Deployed model: **SetFit (silver)**; teacher: **LLM zero-shot** (Gemini, prompt v2).


## SetFit (silver): 87 errors

| gold       | predicted       |   n | share_of_errors   |
|:-----------|:----------------|----:|:------------------|
| complaint  | feature_request |  15 | 17%               |
| complaint  | bug_report      |  13 | 15%               |
| question   | other           |   8 | 9%                |
| bug_report | question        |   6 | 7%                |
| bug_report | complaint       |   6 | 7%                |
| praise     | complaint       |   6 | 7%                |
| praise     | other           |   4 | 5%                |
| other      | complaint       |   4 | 5%                |
| question   | feature_request |   4 | 5%                |
| bug_report | feature_request |   3 | 3%                |


## LLM zero-shot: 86 errors

| gold            | predicted       |   n | share_of_errors   |
|:----------------|:----------------|----:|:------------------|
| complaint       | feature_request |  16 | 19%               |
| complaint       | bug_report      |  12 | 14%               |
| praise          | complaint       |   7 | 8%                |
| question        | other           |   6 | 7%                |
| question        | feature_request |   6 | 7%                |
| bug_report      | complaint       |   4 | 5%                |
| bug_report      | question        |   4 | 5%                |
| feature_request | complaint       |   4 | 5%                |
| bug_report      | feature_request |   3 | 3%                |
| praise          | other           |   3 | 3%                |


## Shared mistakes

Both systems are wrong on 60 items; on 51 of them they agree on the same wrong label (59% of the deployed model's errors). These are the candidates for genuinely ambiguous items or annotator conventions.

| gold       | both predicted   |   n |
|:-----------|:-----------------|----:|
| complaint  | feature_request  |   8 |
| complaint  | bug_report       |   8 |
| question   | other            |   6 |
| praise     | complaint        |   5 |
| bug_report | question         |   4 |
| bug_report | complaint        |   3 |
| other      | bug_report       |   2 |
| complaint  | question         |   2 |


## Examples of the deployed model's top confusions


### gold `complaint` → predicted `feature_request` (15)

- *play*, teacher said `feature_request`: stop stopping my music to ask me to rate you!
- *play*, teacher said `complaint`: I know Spotify premium is your business model but that doesn't mean to force your users to buy it . So back to YouTube atleast I can play whichever song I want to listen and allow to listen previous and next song with h…
- *play*, teacher said `bug_report`: I'm a premium user, and it has been a good experience, but lately, I've been having some serious issues with the app . I can't asses my downloaded songs or anything on Spotify, if my data is off and I click on Spotify a…
- *play*, teacher said `complaint`: This app is average at its best it could be much better.. The search for artists is good but they don't have every album by the artists on this app.. why? Also I think there is hardly any perks or promotions available f…


### gold `complaint` → predicted `bug_report` (13)

- *play*, teacher said `bug_report`: Unreliable when offline. I pay for premium family and we went camping for just over a week. Downloaded playlists, podcasts and audiobooks worked for less than a day across different phones.once it stopped loading, that …
- *play*, teacher said `complaint`: AI MUSIC IS DESTROYING THIS APP, sick of fake artist's been shown to me music is art and expression! AI is crashing and ruining it all
- *play*, teacher said `complaint`: This is very good app but now I don't know that it was showing on only on my phone but according to notice now I can not play song according to me and if I want to play song according to me so I have to take premium of …
- *play*, teacher said `bug_report`: the problem I'm having is that spotify keeps moving my saved tracks from SD card to internal storage. Then when I try to move it back, the app crashes and stops working permanently. I then have to reinstall the app. But…


### gold `question` → predicted `other` (8)

- *reddit_comment*, teacher said `other`: It has the same layouts as the Mix feature in your own playlists. Did you click that by any chance?
- *reddit_comment*, teacher said `question`: Are all those songs downloaded to your phone to be able to listen to them offline? Because clearing the cache and not having your entire liked playlist of 10,000+ songs saved into your phone will fix it and have it runn…
- *reddit_post*, teacher said `question`: Am I depressed ! or you can suggest better than this to go more in flow
- *reddit_post*, teacher said `other`: why do some of you hate podcasts on spotify? i love being able to listen to podcasts on here, and i dont know anyone irl witht his problem


### gold `bug_report` → predicted `question` (6)

- *play*, teacher said `bug_report`: Spotify app auto closes when I launch it on my phone. This has been going on for hours and I'm unable to use it. No advisory posted anywhere? Are you fixing this anytime soon???
- *play*, teacher said `bug_report`: I want to cancel my Spotify Premium but I do not see the option to do so, neither on Spotify nor on Google Playstore. Kindly look into it and I look forward to finding a solution on this problem. I hope it is a technica…
- *reddit_post*, teacher said `question`: Search bar not working No matter what I search this is what I see, I have tried reinstalling, clearing cache and logging out and back in. Any help would be amazing 🙏🙏
- *reddit_post*, teacher said `question`: Unknown songs in the playlist "On Repeat" This morning I wanted to listen to the playlist of the most listened-to songs on my profile and songs appeared that I've never heard in my life. Is this a glitch?


### gold `bug_report` → predicted `complaint` (6)

- *play*, teacher said `bug_report`: they obviously care more about marketing and cashing a check than fixing any of their glitches. for instance, when you're watching a podcast video, if you exit Spotify or turn your screen, the lip syncing is off. this i…
- *play*, teacher said `complaint`: I get 4 ads in a row after playing one song. After said song is over, I choose a different one, and then I immediately get hit with another ad again happens so very often
- *play*, teacher said `complaint`: I pay for the subscription but the lyrics that you get is all incorrect. this is just an app for music do not pay for it.
- *play*, teacher said `complaint`: it just said it'll give me 30 minutes off interrupted music and then gave me another 2 ads after 1 song ...


### gold `praise` → predicted `complaint` (6)

- *play*, teacher said `complaint`: it was pretty good but it has a lot of features you need premium for. you get 1-4 ads every thirty minutes and you only get a few skips. it isn't the worst I mean it gets the job done, but there's definitely better opti…
- *play*, teacher said `praise`: this is my favorite music app it it is amazing and cool this is all that I have to say oh one more thing there is adds lol ok bye have a great day
- *play*, teacher said `complaint`: not a bad app but not as enjoyable without premium I mean its annoying listening to ads everytime a song finishes not to talk about the amount of skips is kinda low not a bad app I give it 4 stars ⭐⭐⭐⭐
- *play*, teacher said `complaint`: Good solid user interface. Only supports official Spotify music. It doesn't replace YouTube creative music mixes which are readily available on YouTube. I use and pay for subscriptions on both, however still get some in…

