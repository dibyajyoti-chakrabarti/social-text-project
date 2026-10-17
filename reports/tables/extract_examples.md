# Extraction examples

## Gold

> Would love to be able to adjust DB of songs ourselves. Like I said, it would be nice if instead of “volume normalization” that we could just adjust all the tracks either individually or to one DB level. Even with volume normalization it’s highly annoying that some tracks get quieter, some depending …

- annotator: ['adjust db of song']
- rules [correct]: ['adjust db of songs ourselves']
- llm [correct]: ['manual track volume adjustment']

> I love listening to music in this app!! I would give it five but just one complaint..... whenever I loop a song (I'm using Samsung tablet to repeat songs) the song would play as usual, but when it ends, and ad would pop up. after the ad is over the music just... stops.... so then I have to go back t…

- annotator: ['play song after ad']
- rules [correct]: []
- llm [correct]: []

> could do better,not advertising every other song, when you say, enjoy the next 30mins, then. keep interrupting with an add.

- annotator: ['dont interrupt the 30 mins promised']
- rules [wrong]: []
- llm [correct]: ['fewer ads']

> Your new user interface for Android Auto is dumb! Why over complicate it? There was nothing wrong with its interface before. Put it back!

- annotator: ['put back old ui']
- rules [wrong]: []
- llm [correct]: ['bring back old android auto ui']

> Every time when a make a damn playlist, it just shuffles it which I hate. So stop and i'm not paying a damn subscription, just type put what I want cause.

- annotator: ['play what user wants']
- rules [wrong]: []
- llm [partial]: ['disable forced shuffle on playlists', 'on-demand song selection on free tier']

> it asks me money for every thing even to select a song Spotify pls change this

- annotator: ['dont ask for money always']
- rules [wrong]: []
- llm [wrong]: ['on-demand song selection on free tier']

> I started using it on my phone and honestly it felt like everything that had been bottled up was finally being released because this app just has everything u need when u down.but I think that it can do more better in terms of keeping u in the save vibe of songs,so you don't have to worry about the …

- annotator: ['next song of same vibe']
- rules [wrong]: []
- llm [correct]: ['consistent vibe recommendations']

> I wish I could give it a higher rating, but I cant. I add about ten songs to my queue, and it only let's me listen to THREE before it goes "Oh no, now you cant choose your songs, sorry not sorry!" Also, I cant have premium because I cant afford what, 13 dollars a month?! For music?! No. And, only si…

- annotator: ['make the price cheaper']
- rules [wrong]: ['give it higher rating', 'chuck my phone across room']
- llm [correct]: ['remove skip limits', 'cheaper premium plan']

> too many ads, large music selection tho so its gucci 👌. REDUCE PREMIUM PRICES PLSSS

- annotator: ['reduction of premium prices']
- rules [correct]: ['reduce premium prices plsss']
- llm [correct]: ['fewer ads', 'cheaper premium plan']

> I would give it a 5 till the new update because if u don't have money buy premium then you have your music shuffled and I don't like that and it adds more songs to one song u just want to listen to and I don't really like that please change it but besides that it's pretty good music app

- annotator: ['make non-premium experience better']
- rules [wrong]: ['listen to']
- llm [partial]: ['disable forced shuffle on free tier']

> I liked the app until the skip/rewind option disappeared. I mean like why are they not available, initially I thought it's some bug so I reinstalled it. Yet the option doesn't show. Been more than a week now. Without skip/rewind option (even limited times) it's not as fun. I hope that the team will …

- annotator: ['bring back skip/ rewind']
- rules [partial]: ['bring back option']
- llm [correct]: ['bring back skip and rewind buttons']

> Would be cool if they just make it single tap to put it in liked songs and long press and drag to add to the playlist (like how pinterest does it)

- annotator: ['add gimmicks like pintrest']
- rules [wrong]: []
- llm [correct]: ['single tap liked songs', 'drag and drop playlist management']

> 4 stars but not 5. Spotify... Can you help? When I save a song to a playlist, sometimes it doesn't save at all. I have to try and save a song to a playlist a number of times. This is risking songs getting lost and not being saved. I pay for premium, I shouldn't be having to double check that a song …

- annotator: ['make sure a song is actually saved']
- rules [correct]: []
- llm [wrong]: []

> Probably one of the greediest companies currently, free plan is basically unusable, thousands of ads you can't skip, stuck on shuffle, and limited to 7 song skips per hour and 6 song choices per DAY, genuinely awful app, Spotify PLEASE at least let us choose the songs we want to listen to without a …

- annotator: ['lower the price of premium', 'freedom to choose song']
- rules [correct]: ['listen to without limit', 'lower price of membership']
- llm [wrong]: ['remove skip limits', 'on-demand song selection on free tier', 'cheaper premium plan']

> Far too keen to play music I haven't asked for. Stop that. Now.

- annotator: ['stop unsought music playing']
- rules [wrong]: []
- llm [correct]: ['stop playing unrequested music']

## Corpus sample

> app is doing too much, make a lite version bc this is ridiculous, I literally can't do anything in the app

- rules [wrong]: [] (competitor: nan)
- llm [correct]: ['spotify lite app'] (competitor: nan)

> The app and the widgets worked well and then they decided to force me to use the newer, larger widget that never works and crashes constantly. give me back the widgets that work. They replied to do basic troubleshooting, which has already been done and is basic common sense. Their widget is not read…

- rules [correct]: ['widgets that work'] (competitor: nan)
- llm [correct]: ['bring back old widgets'] (competitor: nan)

> I actually really like everything I just want the eat the playlist feature for android. I dont get why I dont have it when I'm paying for premium doesn't make any sense.

- rules [wrong]: [] (competitor: nan)
- llm [correct]: ['eat the playlist for android'] (competitor: nan)

> If I wanted Adele's opinion on how to consume media, I would listen to her awful music. Let me keep my artist shuffle without needing to reset it every 30 minutes. I'm updating because I do still use the app. I just got a notification that my audiobook time was up. I have 20 minutes left in the book…

- rules [partial]: ['keep my artist shuffle without needing to reset', 'half hour of time'] (competitor: nan)
- llm [correct]: ['persistent artist shuffle'] (competitor: nan)

> I wish it had free access to autobooks books for everyone it would make it easier (hope i spelt everything right)

- rules [partial]: ['had free access to autobooks books for everyone', 'make it easier'] (competitor: nan)
- llm [correct]: ['free audiobooks access'] (competitor: nan)

> They really need to notify you about songs getting taken down. I have soooo many songs in my playlists that it just skips over now 😞

- rules [wrong]: [] (competitor: nan)
- llm [correct]: ['removed song notifications'] (competitor: nan)

> Spotify the great collection of music and podcasts, until it ends up pushing gambling and explicit content to your family. Remove the gambling ADs, yesterday preferably. Have you no shame?

- rules [correct]: ['remove gambling ads'] (competitor: nan)
- llm [correct]: ['remove gambling ads'] (competitor: nan)

> the UI has become so cluttered with ads for videos and podcasts that it's getting hard to find any music. I pay for premium and it's gotten so bad I am jumping ship to either YT, Apple, or Deezer. The recommendations for music have gotten way out of whack as well. Spotify also has this thing about p…

- rules [partial]: ['bring back spotify of 5 years ago'] (competitor: YouTube Music, Deezer)
- llm [partial]: ['less cluttered ui', 'hide specific artists'] (competitor: YouTube)

> Put my music playlist into shuffle

- rules [partial]: ['put my music playlist'] (competitor: nan)
- llm [correct]: ['playlist shuffle'] (competitor: nan)

> the new dj mix algorithm needs more adjustment. Always repeating the same songs even if I do things to try to switch it up. don't add to que while using it because it will restart you mix with the same order of songs. I shouldn't feel like I can guess what will play next.

- rules [wrong]: [] (competitor: nan)
- llm [partial]: ['better dj mix algorithm'] (competitor: nan)

> After having the app for a long time, I think its really great! One of the things they could improve on, however is some of the personalization settings, i wish that there was a way to just block explicit podcasts. Other than that, I think it's so much better and has so much more variety than other …

- rules [partial]: ['of things they could improve on', 'that there was way to just block', 'just block explicit podcasts'] (competitor: nan)
- llm [correct]: ['block explicit podcasts'] (competitor: nan)

> If you never stream from india then they know you are not there. If only there was a quick and easy way to connect from a different country to these services.

- rules [correct]: ['connect from different country to these services'] (competitor: nan)
- llm [wrong]: [] (competitor: nan)

> not bad but users should be given some offline songs to listen

- rules [correct]: ['offline songs'] (competitor: nan)
- llm [correct]: ['free offline songs'] (competitor: nan)

> just wish the ads were bit less for those who don't pay any subscription😭

- rules [partial]: ['ads were bit less for those who'] (competitor: nan)
- llm [correct]: ['fewer ads for free tier'] (competitor: nan)

> This is a pretty good music player app but there's so much frickin' ads! You have to PAY premium just to hear songs in your order, move your place in the song, and get unlimited skips. I thought that was the bare minimum an app made for playing music should do!? Also, the song order editor in playli…

- rules [partial]: ['unlimited skips', 'be in playlist'] (competitor: nan)
- llm [correct]: ['fewer ads', 'free song ordering', 'free seeking', 'unlimited skips', 'move multiple playlist songs'] (competitor: nan)

