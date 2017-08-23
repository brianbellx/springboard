Much of the code has been commented out.  This was code that took a long time to execute, but only needed to run once and store the results to disk.  The results of blocks commented out is usually loaded from disk right after the comment block.

dataorg.py, grscrape, and grscrape2 are the data-wrangling code that took the raw data in BX-Books.csv and BX-Book-Ratings.csv to booksclean.csv and ratings.csv.  They do not need to run to see the results of the project.

Usage instructions

bybook.py:

	bookrec('ISBN') -- returns a list of recommendations based on an example book.

byuser.py:
	
	userrecs(userID) -- returns a list of recommendations for a given user

kmeans.py:

	class recommender()
	recommender.SVD(n_components) -- performs decomposition into n_components vectors
	recommender.buildKMeans(n_clusters) -- cluster in to n_clusters
	recommender.GroupRec(groupID) -- give recommendations for the provided cluster
	recommender.visualize() -- show a scatterplot of the clusters

LDA.py:
	displaysim(ldacomp('ISBN')) -- show similar books by lda topic comparison
	displaysim(w2vcomp('ISBN')) -- show similar books by word vector comparison
	displaysim(w2vcomptext(u'text')) -- show books similar to provided unicode text
	
