# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:02:12 2022

@author: "Petalinkar Saša"
"""
import nltk
import time


raw = """1. This is the secret of the Holy Graal, that is the sacred vessel of our Lady the Scarlet Woman, Babalon the Mother of Abominations, the bride of Chaos, that rideth upon our Lord the Beast.

2. Thou shalt drain out thy blood that is thy life into the golden cup of her fornication.

3. Thou shalt mingle thy life with the universal life. Thou shalt keep not back one drop.

4. Then shall thy brain be dumb, and thy heart beat no more, and all thy life shall go from thee; and thou shalt be cast out upon the midden, and the birds of the air shall feast upon thy flesh, and thy bones shall whiten in the sun.

5. Then shall the winds gather themselves together, and bear thee up as it were a little heap of dust in a sheet that hath four corners, and they shall give it unto the guardians of the abyss.

6. And because there is no life therein, the guardians of the abyss shall bid the angels of the winds pass by. And the angels shall lay thy dust in the City of the Pyramids, and the name thereof shall be no more.

7. Now therefore that thou mayest achieve this ritual of the Holy Graal, do thou divest thyself of all thy goods.

8. Thou hast wealth; give it unto them that have need thereof, yet no desire toward it.

9. Thou hast health; slay thyself in the fervour of thine abandonment unto Our Lady. Let thy flesh hang loose upon thy bones, and thine eyes glare with thy quenchless lust unto the Infinite, with thy passion for the Unknown, for Her that is beyond Knowledge the accursed one.

10. Thou hast love; tear thy mother from thine heart, and spit in the face of thy father. Let thy foot trample the belly of thy wife, and let the babe at her breast be the prey of dogs and vultures.

11. For if thou dost not this with thy will, then shall We do this despite thy will. So that thou attain to the Sacrament of the Graal in the Chapel of Abominations.

12. And behold! if by stealth thou keep unto thyself one thought of thine, then shalt thou be cast out into the abyss for ever; and thou shalt be the lonely one, the eater of dung, the afflicted in the Day of Be-with-Us.

13. Yea! verily this is the Truth, this is the Truth, this is the Truth. Unto thee shall be granted joy and health and wealth and wisdom when thou art no longer thou.

14. Then shall every gain be a new sacrament, and it shall not defile thee; thou shalt revel with the wanton in the market-place, and the virgins shall fling roses upon thee, and the merchants bend their knees and bring thee gold and spices. Also young boys shall pour wonderful wines for thee, and the singers and the dancers shall sing and dance for thee.

15. Yet shalt thou not be therein, for thou shalt be forgotten, dust lost in dust.

16. Nor shall the aeon itself avail thee in this; for from the dust shall a white ash be prepared by Hermes the Invisible.

17. And this is the wrath of God, that these things should be thus.

18. And this is the grace of God, that these things should be thus.

19. Wherefore I charge you that ye come unto me in the Beginning; for if ye take but one step in this Path, ye must arrive inevitably at the end thereof.

20. This Path is beyond Life and Death; it is also beyond Love; but that ye know not, for ye know not Love.

21. And the end thereof is known not even unto Our Lady or to the Beast whereon She rideth; nor unto the Virgin her daughter nor unto Chaos her lawful Lord; but unto the Crowned Child is it known? It is not known if it be known.

22. Therefore unto Hadit and unto Nuit be the glory in the End and the Beginning; yea, in the End and the Beginning.

1. Had! The manifestation of Nuit.

2. The unveiling of the company of heaven.

3. Every man and every woman is a star.

4. Every number is infinite; there is no difference.

5. Help me, o warrior lord of Thebes, in my unveiling before the Children of men!

6. Be thou Hadit, my secret centre, my heart & my tongue!

7. Behold! it is revealed by Aiwass the minister of Hoor-paar-kraat.

8. The Khabs is in the Khu, not the Khu in the Khabs.

9. Worship then the Khabs, and behold my light shed over you!

10. Let my servants be few & secret: they shall rule the many & the known.

11. These are fools that men adore; both their Gods & their men are fools.

12. Come forth, o children, under the stars, & take your fill of love!

13. I am above you and in you. My ecstasy is in yours. My joy is to see your joy.

14. Above, the gemmèd azure is The naked splendour of Nuit; She bends in ecstasy to kiss The secret ardours of Hadit. The winged globe, the starry blue, Are mine, O Ankh-af-na-khonsu!

15. Now ye shall know that the chosen priest & apostle of infinite space is the prince-priest the Beast; and in his woman called the Scarlet Woman is all power given. They shall gather my children into their fold: they shall bring the glory of the stars into the hearts of men.

16. For he is ever a sun, and she a moon. But to him is the winged secret flame, and to her the stooping starlight.

17. But ye are not so chosen.

18. Burn upon their brows, o splendrous serpent!

19. O azure-lidded woman, bend upon them!

20. The key of the rituals is in the secret word which I have given unto him.

21. With the God & the Adorer I am nothing: they do not see me. They are as upon the earth; I am Heaven, and there is no other God than me, and my lord Hadit.

22. Now, therefore, I am known to ye by my name Nuit, and to him by a secret name which I will give him when at last he knoweth me. Since I am Infinite Space, and the Infinite Stars thereof, do ye also thus. Bind nothing! Let there be no difference made among you between any one thing & any other thing; for thereby there cometh hurt.

23. But whoso availeth in this, let him be the chief of all!

24. I am Nuit, and my word is six and fifty.

25. Divide, add, multiply, and understand.

26. Then saith the prophet and slave of the beauteous one: Who am I, and what shall be the sign? So she answered him, bending down, a lambent flame of blue, all-touching, all penetrant, her lovely hands upon the black earth, & her lithe body arched for love, and her soft feet not hurting the little flowers: Thou knowest! And the sign shall be my ecstasy, the consciousness of the continuity of existence, the omnipresence of my body.

27. Then the priest answered & said unto the Queen of Space, kissing her lovely brows, and the dew of her light bathing his whole body in a sweet-smelling perfume of sweat: O Nuit, continuous one of Heaven, let it be ever thus; that men speak not of Thee as One but as None; and let them speak not of thee at all, since thou art continuous!

28. None, breathed the light, faint & faery, of the stars, and two.

29. For I am divided for love's sake, for the chance of union.

30. This is the creation of the world, that the pain of division is as nothing, and the joy of dissolution all.

31. For these fools of men and their woes care not thou at all! They feel little; what is, is balanced by weak joys; but ye are my chosen ones.

32. Obey my prophet! follow out the ordeals of my knowledge! seek me only! Then the joys of my love will redeem ye from all pain. This is so: I swear it by the vault of my body; by my sacred heart and tongue; by all I can give, by all I desire of ye all.

33. Then the priest fell into a deep trance or swoon, & said unto the Queen of Heaven; Write unto us the ordeals; write unto us the rituals; write unto us the law!

34. But she said: the ordeals I write not: the rituals shall be half known and half concealed: the Law is for all.

35. This that thou writest is the threefold book of Law.

36. My scribe Ankh-af-na-khonsu, the priest of the princes, shall not in one letter change this book; but lest there be folly, he shall comment thereupon by the wisdom of Ra-Hoor-Khu-it.

37. Also the mantras and spells; the obeah and the wanga; the work of the wand and the work of the sword; these he shall learn and teach.

38. He must teach; but he may make severe the ordeals.

39. The word of the Law is θελημα.

40. Who calls us Thelemites will do no wrong, if he look but close into the word. For there are therein Three Grades, the Hermit, and the Lover, and the man of Earth. Do what thou wilt shall be the whole of the Law.

41. The word of Sin is Restriction. O man! refuse not thy wife, if she will! O lover, if thou wilt, depart! There is no bond that can unite the divided but love: all else is a curse. Accursed! Accursed be it to the aeons! Hell.

42. Let it be that state of manyhood bound and loathing. So with thy all; thou hast no right but to do thy will.

43. Do that, and no other shall say nay.

44. For pure will, unassuaged of purpose, delivered from the lust of result, is every way perfect.

45. The Perfect and the Perfect are one Perfect and not two; nay, are none!

46. Nothing is a secret key of this law. Sixty-one the Jews call it; I call it eight, eighty, four hundred & eighteen.

47. But they have the half: unite by thine art so that all disappear.

48. My prophet is a fool with his one, one, one; are not they the Ox, and none by the Book?

49. Abrogate are all rituals, all ordeals, all words and signs. Ra-Hoor-Khuit hath taken his seat in the East at the Equinox of the Gods; and let Asar be with Isa, who also are one. But they are not of me. Let Asar be the adorant, Isa the sufferer; Hoor in his secret name and splendour is the Lord initiating.

50. There is a word to say about the Hierophantic task. Behold! there are three ordeals in one, and it may be given in three ways. The gross must pass through fire; let the fine be tried in intellect, and the lofty chosen ones in the highest. Thus ye have star & star, system & system; let not one know well the other!

51. There are four gates to one palace; the floor of that palace is of silver and gold; lapis lazuli & jasper are there; and all rare scents; jasmine & rose, and the emblems of death. Let him enter in turn or at once the four gates; let him stand on the floor of the palace. Will he not sink? Amn. Ho! warrior, if thy servant sink? But there are means and means. Be goodly therefore: dress ye all in fine apparel; eat rich foods and drink sweet wines and wines that foam! Also, take your fill and will of love as ye will, when, where and with whom ye will! But always unto me.

52. If this be not aright; if ye confound the space-marks, saying: They are one; or saying, They are many; if the ritual be not ever unto me: then expect the direful judgments of Ra Hoor Khuit!

53. This shall regenerate the world, the little world my sister, my heart & my tongue, unto whom I send this kiss. Also, o scribe and prophet, though thou be of the princes, it shall not assuage thee nor absolve thee. But ecstasy be thine and joy of earth: ever To me! To me!

54. Change not as much as the style of a letter; for behold! thou, o prophet, shalt not behold all these mysteries hidden therein.

55. The child of thy bowels, he shall behold them.

56. Expect him not from the East, nor from the West; for from no expected house cometh that child. Aum! All words are sacred and all prophets true; save only that they understand a little; solve the first half of the equation, leave the second unattacked. But thou hast all in the clear light, and some, though not all, in the dark.

57. Invoke me under my stars! Love is the law, love under will. Nor let the fools mistake love; for there are love and love. There is the dove, and there is the serpent. Choose ye well! He, my prophet, hath chosen, knowing the law of the fortress, and the great mystery of the House of God.

All these old letters of my Book are aright; but צ is not the Star. This also is secret: my prophet shall reveal it to the wise.

58. I give unimaginable joys on earth: certainty, not faith, while in life, upon death; peace unutterable, rest, ecstasy; nor do I demand aught in sacrifice.

59. My incense is of resinous woods & gums; and there is no blood therein: because of my hair the trees of Eternity.

60. My number is 11, as all their numbers who are of us. The Five Pointed Star, with a Circle in the Middle, & the circle is Red. My colour is black to the blind, but the blue & gold are seen of the seeing. Also I have a secret glory for them that love me.

61. But to love me is better than all things: if under the night stars in the desert thou presently burnest mine incense before me, invoking me with a pure heart, and the Serpent flame therein, thou shalt come a little to lie in my bosom. For one kiss wilt thou then be willing to give all; but whoso gives one particle of dust shall lose all in that hour. Ye shall gather goods and store of women and spices; ye shall wear rich jewels; ye shall exceed the nations of the earth in splendour & pride; but always in the love of me, and so shall ye come to my joy. I charge you earnestly to come before me in a single robe, and covered with a rich headdress. I love you! I yearn to you! Pale or purple, veiled or voluptuous, I who am all pleasure and purple, and drunkenness of the innermost sense, desire you. Put on the wings, and arouse the coiled splendour within you: come unto me!

62. At all my meetings with you shall the priestess say -- and her eyes shall burn with desire as she stands bare and rejoicing in my secret temple -- To me! To me! calling forth the flame of the hearts of all in her love-chant.

63. Sing the rapturous love-song unto me! Burn to me perfumes! Wear to me jewels! Drink to me, for I love you! I love you!

64. I am the blue-lidded daughter of Sunset; I am the naked brilliance of the voluptuous night-sky.

65. To me! To me!

66. The Manifestation of Nuit is at an end.

Chapter II
1. Nu! the hiding of Hadit.

2. Come! all ye, and learn the secret that hath not yet been revealed. I, Hadit, am the complement of Nu, my bride. I am not extended, and Khabs is the name of my House.

3. In the sphere I am everywhere the centre, as she, the circumference, is nowhere found.

4. Yet she shall be known & I never.

5. Behold! the rituals of the old time are black. Let the evil ones be cast away; let the good ones be purged by the prophet! Then shall this Knowledge go aright.

6. I am the flame that burns in every heart of man, and in the core of every star. I am Life, and the giver of Life, yet therefore is the knowledge of me the knowledge of death.

7. I am the Magician and the Exorcist. I am the axle of the wheel, and the cube in the circle. "Come unto me" is a foolish word: for it is I that go.

8. Who worshipped Heru-pa-kraath have worshipped me; ill, for I am the worshipper.

9. Remember all ye that existence is pure joy; that all the sorrows are but as shadows; they pass & are done; but there is that which remains.

10. O prophet! thou hast ill will to learn this writing.

11. I see thee hate the hand & the pen; but I am stronger.

12. Because of me in Thee which thou knewest not.

13. for why? Because thou wast the knower, and me.

14. Now let there be a veiling of this shrine: now let the light devour men and eat them up with blindness!

15. For I am perfect, being Not; and my number is nine by the fools; but with the just I am eight, and one in eight: Which is vital, for I am none indeed. The Empress and the King are not of me; for there is a further secret.

16. I am The Empress & the Hierophant. Thus eleven, as my bride is eleven.

17. Hear me, ye people of sighing!

The sorrows of pain and regret
Are left to the dead and the dying,

The folk that not know me as yet.
18. These are dead, these fellows; they feel not. We are not for the poor and sad: the lords of the earth are our kinsfolk.

19. Is a God to live in a dog? No! but the highest are of us. They shall rejoice, our chosen: who sorroweth is not of us.

20. Beauty and strength, leaping laughter and delicious languor, force and fire, are of us.

21. We have nothing with the outcast and the unfit: let them die in their misery. For they feel not. Compassion is the vice of kings: stamp down the wretched & the weak: this is the law of the strong: this is our law and the joy of the world. Think not, o king, upon that lie: That Thou Must Die: verily thou shalt not die, but live. Now let it be understood: If the body of the King dissolve, he shall remain in pure ecstasy for ever. Nuit! Hadit! Ra-Hoor-Khuit! The Sun, Strength & Sight, Light; these are for the servants of the Star & the Snake.

22. I am the Snake that giveth Knowledge & Delight and bright glory, and stir the hearts of men with drunkenness. To worship me take wine and strange drugs whereof I will tell my prophet, & be drunk thereof! They shall not harm ye at all. It is a lie, this folly against self. The exposure of innocence is a lie. Be strong, o man! lust, enjoy all things of sense and rapture: fear not that any God shall deny thee for this.

23. I am alone: there is no God where I am.

24. Behold! these be grave mysteries; for there are also of my friends who be hermits. Now think not to find them in the forest or on the mountain; but in beds of purple, caressed by magnificent beasts of women with large limbs, and fire and light in their eyes, and masses of flaming hair about them; there shall ye find them. Ye shall see them at rule, at victorious armies, at all the joy; and there shall be in them a joy a million times greater than this. Beware lest any force another, King against King! Love one another with burning hearts; on the low men trample in the fierce lust of your pride, in the day of your wrath.

25. Ye are against the people, O my chosen!

26. I am the secret Serpent coiled about to spring: in my coiling there is joy. If I lift up my head, I and my Nuit are one. If I droop down mine head, and shoot forth venom, then is rapture of the earth, and I and the earth are one.

27. There is great danger in me; for who doth not understand these runes shall make a great miss. He shall fall down into the pit called Because, and there he shall perish with the dogs of Reason.

28. Now a curse upon Because and his kin!

29. May Because be accursèd for ever!

30. If Will stops and cries Why, invoking Because, then Will stops & does nought.

31. If Power asks why, then is Power weakness.

32. Also reason is a lie; for there is a factor infinite & unknown; & all their words are skew-wise.

33. Enough of Because! Be he damned for a dog!

34. But ye, o my people, rise up & awake!

35. Let the rituals be rightly performed with joy & beauty!

36. There are rituals of the elements and feasts of the times.

37. A feast for the first night of the Prophet and his Bride!

38. A feast for the three days of the writing of the Book of the Law.

39. A feast for Tahuti and the child of the Prophet--secret, O Prophet!

40. A feast for the Supreme Ritual, and a feast for the Equinox of the Gods.

41. A feast for fire and a feast for water; a feast for life and a greater feast for death!

42. A feast every day in your hearts in the joy of my rapture!

43. A feast every night unto Nu, and the pleasure of uttermost delight!

44. Aye! feast! rejoice! there is no dread hereafter. There is the dissolution, and eternal ecstasy in the kisses of Nu.

45. There is death for the dogs.

46. Dost thou fail? Art thou sorry? Is fear in thine heart?

47. Where I am these are not.

48. Pity not the fallen! I never knew them. I am not for them. I console not: I hate the consoled & the consoler.

49. I am unique & conqueror. I am not of the slaves that perish. Be they damned & dead! Amen. (This is of the 4: there is a fifth who is invisible, & therein am I as a babe in an egg.)

50. Blue am I and gold in the light of my bride: but the red gleam is in my eyes; & my spangles are purple & green.

51. Purple beyond purple: it is the light higher than eyesight.

52. There is a veil: that veil is black. It is the veil of the modest woman; it is the veil of sorrow, & the pall of death: this is none of me. Tear down that lying spectre of the centuries: veil not your vices in virtuous words: these vices are my service; ye do well, & I will reward you here and hereafter.

53. Fear not, o prophet, when these words are said, thou shalt not be sorry. Thou art emphatically my chosen; and blessed are the eyes that thou shalt look upon with gladness. But I will hide thee in a mask of sorrow: they that see thee shall fear thou art fallen: but I lift thee up.

54. Nor shall they who cry aloud their folly that thou meanest nought avail; thou shall reveal it: thou availest: they are the slaves of because: They are not of me. The stops as thou wilt; the letters? change them not in style or value!

55. Thou shalt obtain the order & value of the English Alphabet; thou shalt find new symbols to attribute them unto.

56. Begone! ye mockers; even though ye laugh in my honour ye shall laugh not long: then when ye are sad know that I have forsaken you.

57. He that is righteous shall be righteous still; he that is filthy shall be filthy still.

58. Yea! deem not of change: ye shall be as ye are, & not other. Therefore the kings of the earth shall be Kings for ever: the slaves shall serve. There is none that shall be cast down or lifted up: all is ever as it was. Yet there are masked ones my servants: it may be that yonder beggar is a King. A King may choose his garment as he will: there is no certain test: but a beggar cannot hide his poverty.

59. Beware therefore! Love all, lest perchance is a King concealed! Say you so? Fool! If he be a King, thou canst not hurt him.

60. Therefore strike hard & low, and to hell with them, master!

61. There is a light before thine eyes, o prophet, a light undesired, most desirable.

62. I am uplifted in thine heart; and the kisses of the stars rain hard upon thy body.

63. Thou art exhaust in the voluptuous fullness of the inspiration; the expiration is sweeter than death, more rapid and laughterful than a caress of Hell's own worm.

64. Oh! thou art overcome: we are upon thee; our delight is all over thee: hail! hail: prophet of Nu! prophet of Had! prophet of Ra-Hoor-Khu! Now rejoice! now come in our splendour & rapture! Come in our passionate peace, & write sweet words for the Kings.

65. I am the Master: thou art the Holy Chosen One.

66. Write, & find ecstasy in writing! Work, & be our bed in working! Thrill with the joy of life & death! Ah! thy death shall be lovely: whoso seeth it shall be glad. Thy death shall be the seal of the promise of our age long love. Come! lift up thine heart & rejoice! We are one; we are none.

67. Hold! Hold! Bear up in thy rapture; fall not in swoon of the excellent kisses!

68. Harder! Hold up thyself! Lift thine head! breathe not so deep -- die!

69. Ah! Ah! What do I feel? Is the word exhausted?

70. There is help & hope in other spells. Wisdom says: be strong! Then canst thou bear more joy. Be not animal; refine thy rapture! If thou drink, drink by the eight and ninety rules of art: if thou love, exceed by delicacy; and if thou do aught joyous, let there be subtlety therein!

71. But exceed! exceed!

72. Strive ever to more! and if thou art truly mine -- and doubt it not, an if thou art ever joyous! -- death is the crown of all.

73. Ah! Ah! Death! Death! thou shalt long for death. Death is forbidden, o man, unto thee.

74. The length of thy longing shall be the strength of its glory. He that lives long & desires death much is ever the King among the Kings.

75. Aye! listen to the numbers & the words:

76. 4 6 3 8 A B K 2 4 A L G M O R 3 Y X 24 89 R P S T O V A L. What meaneth this, o prophet? Thou knowest not; nor shalt thou know ever. There cometh one to follow thee: he shall expound it. But remember, o chosen one, to be me; to follow the love of Nu in the star-lit heaven; to look forth upon men, to tell them this glad word.

77. O be thou proud and mighty among men!

78. Lift up thyself! for there is none like unto thee among men or among Gods! Lift up thyself, o my prophet, thy stature shall surpass the stars. They shall worship thy name, foursquare, mystic, wonderful, the number of the man; and the name of thy house 418.

79. The end of the hiding of Hadit; and blessing & worship to the prophet of the lovely Star!

Chapter III
1. Abrahadabra; the reward of Ra Hoor Khut.

2. There is division hither homeward; there is a word not known. Spelling is defunct; all is not aught. Beware! Hold! Raise the spell of Ra-Hoor-Khuit!

3. Now let it be first understood that I am a god of War and of Vengeance. I shall deal hardly with them.

4. Choose ye an island!

5. Fortify it!

6. Dung it about with enginery of war!

7. I will give you a war-engine.

8. With it ye shall smite the peoples; and none shall stand before you.

9. Lurk! Withdraw! Upon them! this is the Law of the Battle of Conquest: thus shall my worship be about my secret house.

10. Get the stele of revealing itself; set it in thy secret temple -- and that temple is already aright disposed -- & it shall be your Kiblah for ever. It shall not fade, but miraculous colour shall come back to it day after day. Close it in locked glass for a proof to the world.

11. This shall be your only proof. I forbid argument. Conquer! That is enough. I will make easy to you the abstruction from the ill-ordered house in the Victorious City. Thou shalt thyself convey it with worship, o prophet, though thou likest it not. Thou shalt have danger & trouble. Ra-Hoor-Khu is with thee. Worship me with fire & blood; worship me with swords & with spears. Let the woman be girt with a sword before me: let blood flow to my name. Trample down the Heathen; be upon them, o warrior, I will give you of their flesh to eat!

12. Sacrifice cattle, little and big: after a child.

13. But not now.

14. Ye shall see that hour, o blessed Beast, and thou the Scarlet Concubine of his desire!

15. Ye shall be sad thereof.

16. Deem not too eagerly to catch the promises; fear not to undergo the curses. Ye, even ye, know not this meaning all.

17. Fear not at all; fear neither men nor Fates, nor gods, nor anything. Money fear not, nor laughter of the folk folly, nor any other power in heaven or upon the earth or under the earth. Nu is your refuge as Hadit your light; and I am the strength, force, vigour, of your arms.

18. Mercy let be off; damn them who pity! Kill and torture; spare not; be upon them!

19. That stele they shall call the Abomination of Desolation; count well its name, & it shall be to you as 718.

20. Why? Because of the fall of Because, that he is not there again.

21. Set up my image in the East: thou shalt buy thee an image which I will show thee, especial, not unlike the one thou knowest. And it shall be suddenly easy for thee to do this.

22. The other images group around me to support me: let all be worshipped, for they shall cluster to exalt me. I am the visible object of worship; the others are secret; for the Beast & his Bride are they: and for the winners of the Ordeal x. What is this? Thou shalt know.

23. For perfume mix meal & honey & thick leavings of red wine: then oil of Abramelin and olive oil, and afterward soften & smooth down with rich fresh blood.

24. The best blood is of the moon, monthly: then the fresh blood of a child, or dropping from the host of heaven: then of enemies; then of the priest or of the worshippers: last of some beast, no matter what.

25. This burn: of this make cakes & eat unto me. This hath also another use; let it be laid before me, and kept thick with perfumes of your orison: it shall become full of beetles as it were and creeping things sacred unto me.

26. These slay, naming your enemies; & they shall fall before you.

27. Also these shall breed lust & power of lust in you at the eating thereof.

28. Also ye shall be strong in war.

29. Moreover, be they long kept, it is better; for they swell with my force. All before me.

30. My altar is of open brass work: burn thereon in silver or gold!

31. There cometh a rich man from the West who shall pour his gold upon thee.

32. From gold forge steel!

33. Be ready to fly or to smite!

34. But your holy place shall be untouched throughout the centuries: though with fire and sword it be burnt down & shattered, yet an invisible house there standeth, and shall stand until the fall of the Great Equinox; when Hrumachis shall arise and the double-wanded one assume my throne and place. Another prophet shall arise, and bring fresh fever from the skies; another woman shall awakethe lust & worship of the Snake; another soul of God and beast shall mingle in the globed priest; another sacrifice shall stain the tomb; another king shall reign; and blessing no longer be poured To the Hawk-headed mystical Lord!

35. The half of the word of Heru-ra-ha, called Hoor-pa-kraat and Ra-Hoor-Khut.

36. Then said the prophet unto the God:

37. I adore thee in the song -- I am the Lord of Thebes, and I The inspired forth-speaker of Mentu; For me unveils the veiled sky, The self-slain Ankh-af-na-khonsu Whose words are truth. I invoke, I greet Thy presence, O Ra-Hoor-Khuit!

Unity uttermost showed! I adore the might of Thy breath, Supreme and terrible God, Who makest the gods and death To tremble before Thee: -- I, I adore thee!

Appear on the throne of Ra! Open the ways of the Khu! Lighten the ways of the Ka! The ways of the Khabs run through To stir me or still me! Aum! let it fill me!

38. So that thy light is in me; & its red flame is as a sword in my hand to push thy order. There is a secret door that I shall make to establish thy way in all the quarters, (these are the adorations, as thou hast written), as it is said:

The light is mine; its rays consume Me: I have made a secret door Into the House of Ra and Tum, Of Khephra and of Ahathoor. I am thy Theban, O Mentu, The prophet Ankh-af-na-khonsu!

By Bes-na-Maut my breast I beat; By wise Ta-Nech I weave my spell. Show thy star-splendour, O Nuit! Bid me within thine House to dwell, O winged snake of light, Hadit! Abide with me, Ra-Hoor-Khuit!

39. All this and a book to say how thou didst come hither and a reproduction of this ink and paper for ever -- for in it is the word secret & not only in the English -- and thy comment upon this the Book of the Law shall be printed beautifully in red ink and black upon beautiful paper made by hand; and to each man and woman that thou meetest, were it but to dine or to drink at them, it is the Law to give. Then they shall chance to abide in this bliss or no; it is no odds. Do this quickly!

40. But the work of the comment? That is easy; and Hadit burning in thy heart shall make swift and secure thy pen.

41. Establish at thy Kaaba a clerk-house: all must be done well and with business way.

42. The ordeals thou shalt oversee thyself, save only the blind ones. Refuse none, but thou shalt know & destroy the traitors. I am Ra-Hoor-Khuit; and I am powerful to protect my servant. Success is thy proof: argue not; convert not; talk not over much! Them that seek to entrap thee, to overthrow thee, them attack without pity or quarter; & destroy them utterly. Swift as a trodden serpent turn and strike! Be thou yet deadlier than he! Drag down their souls to awful torment: laugh at their fear: spit upon them!

43. Let the Scarlet Woman beware! If pity and compassion and tenderness visit her heart; if she leave my work to toy with old sweetnesses; then shall my vengeance be known. I will slay me her child: I will alienate her heart: I will cast her out from men: as a shrinking and despised harlot shall she crawl through dusk wet streets, and die cold and an-hungered.

44. But let her raise herself in pride! Let her follow me in my way! Let her work the work of wickedness! Let her kill her heart! Let her be loud and adulterous! Let her be covered with jewels, and rich garments, and let her be shameless before all men!

45. Then will I lift her to pinnacles of power: then will I breed from her a child mightier than all the kings of the earth. I will fill her with joy: with my force shall she see & strike at the worship of Nu: she shall achieve Hadit.

46. I am the warrior Lord of the Forties: the Eighties cower before me, & are abased. I will bring you to victory & joy: I will be at your arms in battle & ye shall delight to slay. Success is your proof; courage is your armour; go on, go on, in my strength; & ye shall turn not back for any!

47. This book shall be translated into all tongues: but always with the original in the writing of the Beast; for in the chance shape of the letters and their position to one another: in these are mysteries that no Beast shall divine. Let him not seek to try: but one cometh after him, whence I say not, who shall discover the Key of it all. Then this line drawn is a key: then this circle squared in its failure is a key also. And Abrahadabra. It shall be his child & that strangely. Let him not seek after this; for thereby alone can he fall from it.

48. Now this mystery of the letters is done, and I want to go on to the holier place.

49. I am in a secret fourfold word, the blasphemy against all gods of men.

50. Curse them! Curse them! Curse them!

51. With my Hawk's head I peck at the eyes of Jesus as he hangs upon the cross.

52. I flap my wings in the face of Mohammed & blind him.

53. With my claws I tear out the flesh of the Indian and the Buddhist, Mongol and Din.

54. Bahlasti! Ompehda! I spit on your crapulous creeds.

55. Let Mary inviolate be torn upon wheels: for her sake let all chaste women be utterly despised among you!

56. Also for beauty's sake and love's!

57. Despise also all cowards; professional soldiers who dare not fight, but play; all fools despise!

58. But the keen and the proud, the royal and the lofty; ye are brothers!

59. As brothers fight ye!

60. There is no law beyond Do what thou wilt.

61. There is an end of the word of the God enthroned in Ra's seat, lightening the girders of the soul.

62. To Me do ye reverence! to me come ye through tribulation of ordeal, which is bliss.

63. The fool readeth this Book of the Law, and its comment; & he understandeth it not.

64. Let him come through the first ordeal, & it will be to him as silver.

65. Through the second, gold.

66. Through the third, stones of precious water.

67. Through the fourth, ultimate sparks of the intimate fire.

68. Yet to all it shall seem beautiful. Its enemies who say not so, are mere liars.

69. There is success.

70. I am the Hawk-Headed Lord of Silence & of Strength; my nemyss shrouds the night-blue sky.

71. Hail! ye twin warriors about the pillars of the world! for your time is nigh at hand.

72. I am the Lord of the Double Wand of Power; the wand of the Force of Coph Nia--but my left hand is empty, for I have crushed an Universe; & nought remains.

73. Paste the sheets from right to left and from top to bottom: then behold!

74. There is a splendour in my name hidden and glorious, as the sun of midnight is ever the son.

75. The ending of the words is the Word Abrahadabra.


The Book of the Law is Written and Concealed.

Aum. Ha.

THE COMMENT
Do what thou wilt shall be the whole of the Law.

The study of this Book is forbidden. It is wise to destroy this copy after the first reading.

Whosoever disregards this does so at his own risk and peril. These are most dire.

Those who discuss the contents of this Book are to be shunned by all, as centres of pestilence.

All questions of the Law are to be decided only by appeal to my writings, each for himself.

There is no law beyond Do what thou wilt.

Love is the law, love under will.

The priest of the princes,

Ankh-f-n-khonsu

1. Apep deifieth Asar.

2. Let excellent virgins evoke rejoicing, son of Night!

3. This is the book of the most secret cult of the Ruby Star. It shall be given to none, save to the shameless in deed as in word.

4. No man shall understand this writing—it is too subtle for the sons of men.

5. If the Ruby Star have shed its blood upon thee; if in the season of the moon thou hast invoked by the Iod and the Pe, then mayest thou partake of this most secret sacrament.

6. One shall instruct another, with no care for the matters of men's thought.

7. There shall be a fair altar in the midst, extended upon a black stone.

8. At the head of the altar gold, and twin images in green of the Master.

9. In the midst a cup of green wine.

10. At the foot the Star of Ruby.

11. The altar shall be entirely bare.

12. First, the ritual of the Flaming Star.

13. Next, the ritual of the Seal. 14. Next, the infernal adorations of OAI

Mu pa telai,
Tu wa melai
a, a, a.
Tu fu tulu!
Tu fu tulu
Pa, Sa, Ga.
Qwi Mu telai
Ya Pu melai;
u, u, u.
'Se gu malai;
Pe fu telai,
Fu tu lu.
O chi balae
Wa pa malae:—
Ut! Ut! Ut!
Ge; fu latrai,
Le fu malai
Kut! Hut! Nut!
Al OAI
Rel moai
Ti—Ti—Ti!
Wa la pelai
Tu fu latai
Wi, Ni, Bi.
15. Also thou shalt excite the wheels with the five wounds and the five wounds.

16. Then thou shalt excite the wheels with the two and the third in the midst; even Saturn and Jupiter, Sun and Moon, Mars and Venus, and Mercury.

17. Then the five—and the sixth.

18. Also the altar shall fume before the master with incense that hath no smoke.

19. That which is to be denied shall be denied; that which is to be trampled shall be trampled; that which is to be spat upon shall be spat upon.

20. These things shall be burnt in the outer fire.

21. Then again the master shall speak as he will soft words, and with music and what else he will bring forward the Victim.

22. Also he shall slay a young child upon the altar, and the blood shall cover the altar with perfume as of roses.

23. Then shall the master appear as He should appear—in His glory.

24. He shall stretch himself upon the altar, and awake it into life, and into death.

25. (For so we conceal that life which is beyond.)

26. The temple shall be darkened, save for the fire and the lamp of the altar.

27. There shall he kindle a great fire and a devouring.

28. Also he shall smite the altar with his scourge, and blood shall flow therefrom.

29. Also he shall have made roses bloom thereon.

30. In the end he shall offer up the Vast Sacrifice, at the moment when the God licks up the flame upon the altar.

31. All these things shalt thou perform strictly, observing the time.

32. And the Beloved shall abide with Thee.

33. Thou shalt not disclose the interior world of this rite unto any one: therefore have I written it in symbols that cannot be understood.

34. I who reveal the ritual am IAO and OAI; the Right and the Averse.

35. These are alike unto me.

36. Now the Veil of this operation is called Shame, and the Glory abideth within.

37. Thou shalt comfort the heart of the secret stone with the warm blood. Thou shalt make a subtle decoction of delight, and the Watchers shall drink thereof.

38. I, Apep the Serpent, am the heart of IAO. Isis shall await Asar, and I in the midst.

39. Also the Priestess shall seek another altar, and perform my ceremonies thereon.

40. There shall be no hymn nor dithyramb in my praise and the praise of the rite, seeing that it is utterly beyond.

41. Thou shalt assure thyself of the stability of the altar.

42. In this rite thou shalt be alone.

43. I will give thee another ceremony whereby many shall rejoice.

44. Before all let the Oath be taken firmly as thou raisest up the altar from the black earth.

45. In the words that Thou knowest.

46. For I also swear unto thee by my body and soul that shall never be parted in sunder that I dwell within thee coiled and ready to spring.

47. I will give thee the kingdoms of the earth, O thou Who hast mastered the kingdoms of the East and of the West.

48. I am Apep, O thou slain One. Thou shalt slay thyself upon mine altar: I will have thy blood to drink.

49. For I am a mighty vampire, and my children shall suck up the wine of the earth which is blood.

50. Thou shalt replenish thy veins from the chalice of heaven.

51. Thou shalt be secret, a fear to the world.

52. Thou shalt be exalted, and none shall see thee; exalted, and none shall suspect thee.

53. For there are two glories diverse, and thou who hast won the first shalt enjoy the second.

54. I leap with joy within thee; my head is arisen to strike.

55. O the lust, the sheer rapture, of the life of the snake in the spine!

56. Mightier than God or man, I am in them, and pervade them.

57. Follow out these my words.

58. Fear nothing.

Fear nothing.
Fear nothing.
59. For I am nothing, and me thou shalt fear, O my virgin, my prophet within whose bowels I rejoice.

60. Thou shalt fear with the fear of love: I will overcome thee.

61. Thou shalt be very nigh to death.

62. But I will overcome thee; the New Life shall illumine thee with the Light that is beyond the Stars.

63. Thinkest thou? I, the force that have created all, am not to be despised.

64. And I will slay thee in my lust.

65. Thou shalt scream with the joy and the pain and the fear and the love—so that the ΛΟΓΟΣ of a new God leaps out among the Stars.

66. There shall be no sound heard but this thy lion-roar of rapture; yea, this thy lion-roar of rapture.
"""

rs = time.time()
tokens = nltk.word_tokenize(raw , preserve_line=True)
text = nltk.Text(tokens)
seed = ["Holy Graal"]
text.generate(40,text_seed=(seed),random_seed=rs)
# text.concordance("Graal")
# text.similar("Graal")