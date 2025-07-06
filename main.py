# pip install -U langchain-google-genai


from Agents import QuestionGenerator, Summarizer, QAAgent, Judge

def run_summarization_workflow(query: str, article: str, max_iterations: int = 5):
    question_gen = QuestionGenerator()
    summarizer = Summarizer()
    qa_agent = QAAgent()
    judge_agent = Judge()

    questions = question_gen.run(query=query, article=article)
    current_summary = ""
    sections_to_highlight = [] # For initial run, empty

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # 2. Summarizer
        current_summary = summarizer.run(article=article, sections=sections_to_highlight)
        print("Generated Summary (this iter):")
        print(current_summary)

        # 3. QA
        qa_pairs = qa_agent.run(questions=questions, summary=current_summary)
        print("QA Pairs based on Summary (this iter):")
        for q, a in qa_pairs:
            print(f"Q: {q}\nA: {a}")

        # 4. Judge
        needs_iteration, missing_topics = judge_agent.run(
            article=article,
            summary=current_summary,
            qa_pairs=qa_pairs
        )

        if not needs_iteration:
            print("\nJudge satisfied! Summary is comprehensive.")
            return current_summary, iteration + 1
        else:
            print(f"\nJudge found missing topics. Needs another iteration. Missing topics: {missing_topics}")
            sections_to_highlight = missing_topics # Pass missing topics back for next summarization

    print(f"\nMax iterations ({max_iterations}) reached. Returning current summary.")
    return current_summary, max_iterations

# def run_pipeline(query: str, article: str, max_iter: int = 3) -> str:
#     qg = QuestionGenerator()
#     summarizer = Summarizer()
#     qa_agent = QAAgent()
#     judge = Judge()
#     sections: list[str] = []
#
#     for _ in range(max_iter):
#         questions = qg.run(query, article)
#         summary = summarizer.run(article, sections)
#         qa_pairs = qa_agent.run(questions, summary)
#         continue_loop, missing_topics = judge.run(article, summary, qa_pairs)
#         if not continue_loop:
#             return summary
#         sections.extend(missing_topics)

    return summary

if __name__ == '__main__':
    # sample_article = """
    #     Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem-solving".
    #
    #     AI applications include advanced web search engines (e.g., Google Search), recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative tools (ChatGPT and Midjourney), and competing at the highest level in strategic game systems (such as chess and Go).
    #     """

    sample_article = """
    The Hebrew University of Jerusalem (HUJI; Hebrew: הַאוּנִיבֶרְסִיטָה הַעִבְרִית בִּירוּשָׁלַיִם) is an Israeli public research university based in Jerusalem. Co-founded by Albert Einstein and Chaim Weizmann in July 1918, the public university officially opened on 1 April 1925. It is the second-oldest Israeli university, having been founded 30 years before the establishment of the State of Israel but six years after the older Technion university. The HUJI has three campuses in Jerusalem: one in Rehovot, one in Rishon LeZion and one in Eilat. Until 2023, the world's largest library for Jewish studies—the National Library of Israel—was located on its Edmond J. Safra campus in the Givat Ram neighbourhood of Jerusalem.


The university has five affiliated teaching hospitals (including the Hadassah Medical Center), seven faculties, more than 100 research centers, and 315 academic departments. As of 2018, one-third of all the doctoral candidates in Israel were studying at the HUJI.


Among its first board of governors were Sigmund Freud and Martin Buber. Four of Israel's prime ministers are alumni of the university. As of 2018, 15 Nobel Prize winners (8 alumni and teachers), two Fields Medalists (one alumnus), and three Turing Award winners have been affiliated with the HUJI. It is ranked as the 77th best university in the world.


History

A vision of the Zionist movement was the establishment of a Jewish university in the Land of Israel. Founding a university was proposed as far back as 1884 in the Kattowitz (Katowice) conference of the Hovevei Zion society, and by Hermann Schapira at the First Zionist Congress of 1897.


The cornerstone for the university was laid on 24 July 1918. Seven years later, on 1 April 1925, the Hebrew University campus on Mount Scopus was opened at a gala ceremony attended by the leaders of the Jewish world, distinguished scholars and public figures, and British dignitaries, including the Earl of Balfour, Viscount Allenby, Winston Churchill and Sir Herbert Samuel. The university's first chancellor was Judah Magnes, who led the school as chancellor from 1924 to 1935. In 1935 to 1948 he led the school as president.


One of the most controversial issues during the conceptualization of the university regarded its future official language. Whereas one side, the so-called "Germanists", proposed a combination of German and Arabic for all non-Jewish subjects, the other side opted for the general use of Hebrew. The former party was afraid the very recent Modern Hebrew might not yet allow high-level academic discussions since it still suffered from a lack of specific technical terms in non-religious contexts. Although this concern can not simply be dismissed as unreasonable, the representatives of this position underestimated the symbolic significance of Hebrew for many Jews, not least of all for those outside the academia. Therefore, they were not able to prevail in the discussion and had to give in to the decision that the new university would be an explicitly Hebrew one. The question, what would define the specific Hebrew character of the university did not only regard the choice of an official language but also organizational aspects, as for example the establishment of departments and the definition of their respective research areas, and the outline of its overall academic profile. Therefore, in 1919, Shmaryahu Levin inquired a number of prominent Jewish European scholars about their opinions on the subject. One of the respondents was Ignaz Goldziher whose proposals were at least partly implemented: oriental languages, Jewish literature, and archaeology were among the first subjects studied at the university.


By 1947, the university had become a large research and teaching institution. Sir Leon Simon was Acting President from 1948 to 1949, and he was succeeded as president by Professor Selig Brodetsky, who served from 1949 to 1952. Plans for a medical school were approved in May 1949, and in November 1949, a faculty of law was inaugurated. In 1952, it was announced that the agricultural institute founded by the university in 1940 would become a full-fledged faculty.


During the 1948 Arab–Israeli War, attacks were carried out against convoys moving between the Israeli-controlled section of Jerusalem and the university. The leader of the Arab forces in Jerusalem, Abdul Kader Husseini, threatened military action against the university Hadassah Hospital "if the Jews continued to use them as bases for attacks." After the April 1948 Hadassah medical convoy massacre, in which 79 Jews, including doctors and nurses, were killed, the Mount Scopus campus was cut off from Jerusalem. British soldier Jack Churchill coordinated the evacuation of 700 Jewish doctors, students and patients from the hospital.


When the Jordanian government denied Israeli access to Mount Scopus, a new campus was built at Givat Ram in western Jerusalem and completed in 1958. In the interim, classes were held in 40 different buildings around the city. Benjamin Mazar was President of the university from 1953 to 1961, Giulio Racah was Acting President from 1961 to 1962, and Eliahu Eilat was president from 1962 to 1968.


The Terra Santa building in Rehavia, rented from the Franciscan Custodians of the Latin Holy Places, was also used for this purpose. A few years later, together with the Hadassah Medical Organization, a medical science campus was built in the south-west Jerusalem neighborhood of Ein Kerem.


By the beginning of 1967, the students numbered 12,500, spread among the two campuses in Jerusalem and the agricultural faculty in Rehovot. After the unification of Jerusalem, following the Six-Day War of June 1967, the university was able to return to Mount Scopus, which was rebuilt. According to the Applied Research Institute–Jerusalem, Israel confiscated 568 Dunams of land from the Palestinian village of Isawiya for the Hebrew University in 1968. In 1981 the construction work was completed, and Mount Scopus again became the main campus of the university. Avraham Harman was President of the university from 1968 to 1983, Don Patinkin from 1983 to 1986, Amnon Pazy from 1986 to 1990, Yoram Ben-Porat from 1990 to 1992, Hanoch Gutfreund from 1992 to 1997, and Menachem Magidor from 1997 to 2009.


On 31 July 2002, a member of a terrorist cell detonated a bomb during lunch hour at the university's "Frank Sinatra" cafeteria when it was crowded with staff and students. Nine people—five Israelis, three Americans, and one dual French-American citizen—were murdered and more than 70 wounded. World leaders, including Kofi Annan, President George W. Bush, and the EU's High Representative Javier Solana issued statements of condemnation.


Menachem Ben-Sasson was President of the university from 2009 to 2017, succeeded by Asher Cohen in 2017.


In 2017 the Hebrew University of Jerusalem launched a marijuana research center, intended to "conduct and coordinate research on cannabis and its biological effects with an eye toward commercial applications."


Campuses

Mount Scopus

Mount Scopus (Hebrew: Har HaTzofim הר הצופים), in the north-eastern part of Jerusalem, is home to the main campus, which contains the Faculties of Humanities, Social Sciences, Law, Jerusalem School of Business Administration, Seymour Fox School of Education, Baerwald School of Social Work, Harry S. Truman Research Institute for the Advancement of Peace, Rothberg International School, and the Mandel Institute of Jewish Studies.


The Rothberg International School features secular studies and Jewish/Israeli studies. Included for foreign students is also a mandatory Ulpan program for Hebrew language study which includes a mandatory course in Israeli culture and customs. All Rothberg Ulpan classes are taught by Israeli natives. However, many other classes at the Rothberg School are taught by Jewish immigrants to Israel.


The land on Mt. Scopus was purchased before World War I from Sir John Gray-Hill, along with the Gray-Hill mansion. The master plan for the university was designed by Patrick Geddes and his son-in-law, Frank Mears in December 1919. Only two buildings of this original scheme were built: the David Wolffsohn University and National Library, and the Mathematics Institute, with the Physics Institute probably being built to the designs of their Jerusalem-based partner, Benjamin Chaikin.


Housing for students at Hebrew University who live on Mount Scopus is located at the three dormitories located near the university. These are the Maiersdorf (מאירסדורף) dormitories, the Bronfman (ברונפמן) dormitories, and the Kfar HaStudentim (כפר הסטודנטים, Student Village).


Nearby is the Nicanor Cave, an ancient cave that was planned to be a national pantheon.


Paul Baerwald School of Social Work and Social Welfare

The first Bachelor of Social Work in Israel, the school was founded in 1958.  The school was named after Paul Baerwald, a leader of the Jewish Distribution Committee (JDC). The JDC was an initial funder of the school along with the Ministry of Welfare and the Tel Aviv Municipality. A new self-standing building was dedicated on the Givat Ram university campus in April 1967. The school has been called "the leader in training and research in the fields of social work and social policy." It has been ranked the highest rated school of social work in Israel.  The Master of Social Work was introduced in 1970.  The school houses the Israel Gerontological Data Center, Nevet- Greenhouse of Context-Informed Research and Training for Children in Need, The Center for the Study of Philanthropy in Israel, The Resilience Research Group.


Edmond J. Safra, Givat Ram

The Givat Ram campus (renamed after Edmond Safra in 2005) is the home of the Faculty of Science including the Einstein Institute of Mathematics; the Alexander Silberman Institute of Life Sciences; the Edmond and Lily Safra Center for Brain Sciences; the Israel Institute for Advanced Studies, the Center for the Study of Rationality, as well as the National Library of Israel, (JNUL).


Ein Kerem

The Faculties of Medicine and Dental Medicine and the Institute For Medical Research, Israel-Canada (IMRIC) are located at the south-western Jerusalem Ein Kerem campus alongside the Hadassah-University Medical Center.


Rehovot

The Robert H. Smith Faculty of Agriculture, Food and Environment and the Koret School of Veterinary Medicine are located in the city of Rehovot in the coastal plain. The Faculty was established in 1942 and the School of Veterinary Medicine opened in 1985. These are the only institutions of higher learning in Israel that offer both teaching and research programs in their respective fields. The Faculty is a member of the Euroleague for Life Sciences.


Libraries

The Hebrew University libraries and their web catalogs can be accessed through the HUJI Library Authority portal.


Jewish National and University Library

The Jewish National and University Library is the central and largest library of the Hebrew University and one of the most impressive book and manuscript collections in the world. It is also the oldest section of the university. Founded in 1892 as a world center for the preservation of books relating to Jewish thought and culture, it assumed the additional functions of a general university library in 1920. Its collections of Hebraica and Judaica are the largest in the world. It houses all materials published in Israel, and attempts to acquire all materials published in the world related to the country. It possesses over five million books and thousands of items in special sections, many of which are unique. Among these are the Albert Einstein Archives, Hebrew manuscripts department, Eran Laor map collection, Edelstein science collection, Gershom Scholem collection, and a collection of Maimonides' manuscripts and early writings.


In his will, Albert Einstein left the Hebrew University his personal papers and the copyright to them. The Albert Einstein Archives contain some 55,000 items.  In March 2012 the university announced that it had digitised the entire archive, and was planning to make it more accessible online.  Included in the collection are his personal notes, love letters to various women, including the woman who would become his second wife, Elsa.


Subject-based libraries

In addition to the National Library, the Hebrew University operates subject-based libraries on its campuses, among them the Avraham Harman Science Library, Safra, Givat Ram; Mathematics and Computer Science Library, Safra, Givat Ram; Earth Sciences Library, Safra, Givat Ram; Muriel and Philip I. Berman National Medical Library, Ein Kerem; Central Library of Agricultural Science, Rehovot; Bloomfield Library for the Humanities and Social Sciences, Mt. Scopus; Bernard G. Segal Law Library Center, Mt. Scopus; Emery and Claire Yass Library of the Institute of Archaeology, Mt. Scopus; Moses Leavitt Library of Social Work, Mt. Scopus; Zalman Aranne Central Education Library, Mt. Scopus; Library of the Rothberg School for International Students, Mt. Scopus; Roberta and Stanley Bogen Library of the Harry S. Truman Research Institute for the Advancement of Peace, Mt. Scopus; and the Steven Spielberg Jewish Film Archive.


Rankings

According to the Academic Ranking of World Universities, the Hebrew University is the top university in Israel, overall between 101st and 150th best university in the world, between 301st and 400th in physics, between 201st and 300th in computer science, and between 51st and 75th in business/economics.


In 2021, Shanghai Ranking and the Center for World University Rankings ranked the Hebrew University 1st in Israel in its World University Rankings (90th according to Shanghai Ranking and 64th in the world according to the Center for World University Rankings).


The Hebrew University consistently ranks as Israel's best university in mathematics, and among the best worldwide. It was ranked as the 11th best institution in mathematics worldwide in 2017, 19th best in 2018, 21st best in 2019, and 25th best in 2020.


Friends of the University

The university has an international Society of Friends organizations covering more than 25 countries. Canadian Friends of the Hebrew University of Jerusalem (CFHU), founded in 1944 by Canadian philanthropist Allan Bronfman, promotes awareness, leadership and financial support for The Hebrew University of Jerusalem. CFHU facilitates academic and research partnerships between Canada and Israel as well as establishing scholarships, supporting research, cultivating student and faculty exchanges and recruiting Canadian students to attend the Rothberg International School.  CFHU has chapters in Montreal, Ottawa, Toronto, Winnipeg, Edmonton, Calgary and Vancouver.


The American Friends of the Hebrew University (AFHU) is a not-for-profit 501(c)3 organization that provides programs, events and fundraising activities in support of the university. It was founded by the American philanthropist, Felix M. Warburg in 1925. Supported by its founder, Stephen Floersheimer, and headed by Eran Razin, Floersheimer Studies is a singular program, publishing studies in the field of society, governance and space in Israel. It was established in 2007 replacing the Floersheimer Institute for Policy Studies of 1991.


Faculty

Publications

Institute of Archaeology, Mt. Scopus

Notable alumni

Major award laureates

Political leaders

By profession

Yissum

Yissum Research Development Company is the university's technology transfer company, founded in 1964. Yissum handles all licenses and patents of the researchers and employees of the Hebrew University. Since its formation Yissum has founded more than 80 spin-off companies such as: Mobileye, BriefCam, HumanEyes, OrCam, ExLibris, BioCancell, NewStem and many more. Yissum is led by Yaacov Michlin and other leaders in the business industry such as: Tamir Huberman, Dov Reichman, Shoshi Keinan, Ariela Markel and Michal Levy. Yissum is also a member of ITTN (Israel Technology Transfer Organization).

    """
    sample_query = "What is in the Givat-Ram / Safra campus?"

    final_summary, num_iters = run_summarization_workflow(
        query=sample_query,
        article=sample_article,
        max_iterations=3  # Set your desired max iterations here
    )
    print("\nFinal Summary after workflow:")
    print(final_summary)
    print(f"Workflow completed in {num_iters} iterations.")

# load_dotenv()
    #
    # user_query = "Explain large language models in one sentence."
    # # article_text = "..."  # load or fetch your article here
    # article_text = "As teachers, it is important to know exactly where your students are at in their learning, while at the same time, identifying children who are at risk of falling behind. For students right at the beginning of their schooling, it is important this happens as early as possible, so that teachers can address each student’s individual learning needs. The Australian Early Development Census (AECD) provides some rich data on this and every 3 years, they publish fresh findings that show exactly how students are doing in 5 key domains. The domains are proven to positively impact mental health, wellbeing and educational outcomes in later life. In this article, we look at the results to come from the 2024 study, what this means for teachers, and uncover opportunities for further action. Research shows that when children thrive in their early years, they have a strong foundation for lifelong learning, health, development and wellbeing."
    #
    # final_summary = run_pipeline(user_query, article_text)
    #
    # print(final_summary)

    #
    # # # deal with getting input from user
    #
    # # Initialize the Chat Model (for conversational use cases)
    # # You can specify different Gemini models here, e.g., "gemini-2.5-flash", "gemini-1.5-pro", etc.
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17") # RPM is high
    #
    # # Example text invocation
    # response = llm.invoke("Explain large language models in one sentence.")
    # print(response.content)

    # Example for multimodal input (if using a vision-capable model like gemini-pro-vision or relevant flash models)
    # message = HumanMessage(
    #     content=[
    #         {"type": "text", "text": "What's in this image?"},
    #         {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
    #     ]
    # )
    # response = llm.invoke([message])
    # print(response.content)







# from langchain import ChatOpenAI
# from dotenv import load_dotenv
# import os
#
#
# load_dotenv()
# llm = ChatOpenAI(
#     model="pgt-4-turbo",
#     api_key=os.getenv(API_OAI),
#     temperature=os.getenv("TEMPERATURE")
#     )
#
# response = llm.invoke("is a cucumber more long than green?")
# print(response)
