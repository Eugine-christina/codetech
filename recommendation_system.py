import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

class BookRecommendationSystem:
    def __init__(self):
        self.books_df = None
        self.ratings_df = None
        self.users_df = None
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.svd_model = None
        
    def load_data(self):
        """Load the book recommendation dataset"""
        try:
            self.books_df = pd.read_csv('Book reviews/BX_Books.csv', encoding='latin-1', sep=';', on_bad_lines='skip')
            self.ratings_df = pd.read_csv('Book reviews/BX-Book-Ratings.csv', encoding='latin-1', sep=';', on_bad_lines='skip')
            self.users_df = pd.read_csv('Book reviews/BX-Users.csv', encoding='latin-1', sep=';', on_bad_lines='skip')
            
            print(f"Loaded: {len(self.books_df):,} books, {len(self.ratings_df):,} ratings, {len(self.users_df):,} users")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def preprocess_data(self):
        """Clean and prepare data for modeling"""
        # Filter out zero ratings
        ratings_clean = self.ratings_df[self.ratings_df['Book-Rating'] > 0]
        
        # Merge with book data
        merged_data = pd.merge(ratings_clean, self.books_df, on='ISBN')
        merged_data = merged_data.dropna(subset=['Book-Title', 'Book-Author'])
        
        # Filter for popular books and active users
        book_rating_counts = merged_data['ISBN'].value_counts()
        popular_books = book_rating_counts[book_rating_counts >= 10].index
        filtered_data = merged_data[merged_data['ISBN'].isin(popular_books)]
        
        user_rating_counts = filtered_data['User-ID'].value_counts()
        active_users = user_rating_counts[user_rating_counts >= 5].index
        filtered_data = filtered_data[filtered_data['User-ID'].isin(active_users)]
        
        print(f"Final dataset: {len(filtered_data):,} ratings")
        return filtered_data

    def build_models(self, data):
        """Build collaborative filtering and matrix factorization models"""
        # Collaborative Filtering Model
        print("Building collaborative filtering model...")
        self.user_item_matrix = data.pivot_table(
            index='Book-Title',
            columns='User-ID',
            values='Book-Rating',
            fill_value=0
        )
        
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        print(f"CF model ready with {self.user_item_matrix.shape[0]} books")
        
        # Matrix Factorization Model
        print("Building matrix factorization model...")
        mf_matrix = data.pivot_table(
            index='User-ID',
            columns='ISBN',
            values='Book-Rating',
            fill_value=0
        )
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        self.svd_model.fit(mf_matrix)
        print("MF model ready")

    def find_similar_books(self, book_title, top_n=10):
        """Find similar books using collaborative filtering"""
        if self.similarity_matrix is None:
            return []
            
        # Find closest matching book title
        available_books = list(self.user_item_matrix.index)
        matching_books = [book for book in available_books if book_title.lower() in book.lower()]
        
        if not matching_books:
            print(f"No books found containing '{book_title}'. Showing popular books instead.")
            return self.get_popular_recommendations(top_n)
        
        # Use the first match
        target_book = matching_books[0]
        if len(matching_books) > 1:
            print(f"Found multiple matches. Using: '{target_book}'")
        
        book_idx = available_books.index(target_book)
        similarity_scores = list(enumerate(self.similarity_matrix[book_idx]))
        
        # Sort by similarity score (descending) and exclude the book itself
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        recommendations = []
        for idx, score in sorted_scores:
            similar_book = available_books[idx]
            recommendations.append({
                'title': similar_book,
                'similarity_score': round(score, 4),
                'type': 'Similar Book'
            })
            
        return recommendations

    def get_popular_recommendations(self, top_n=10):
        """Get most popular books based on rating counts"""
        book_ratings = self.ratings_df[self.ratings_df['Book-Rating'] > 0]
        popular_books = book_ratings['ISBN'].value_counts().head(top_n)
        
        recommendations = []
        for isbn, count in popular_books.items():
            book_info = self.books_df[self.books_df['ISBN'] == isbn]
            if not book_info.empty:
                book = book_info.iloc[0]
                recommendations.append({
                    'title': book['Book-Title'],
                    'author': book['Book-Author'],
                    'ratings_count': count,
                    'type': 'Popular Book'
                })
                
        return recommendations

    def search_books(self, query, max_results=10):
        """Search for books by title or author"""
        if not query.strip():
            return []
            
        title_matches = self.books_df[
            self.books_df['Book-Title'].str.contains(query, case=False, na=False)
        ]
        author_matches = self.books_df[
            self.books_df['Book-Author'].str.contains(query, case=False, na=False)
        ]
        
        results = pd.concat([title_matches, author_matches]).drop_duplicates()
        return results.head(max_results)

    def hybrid_recommendations(self, book_title=None, top_n=10):
        """Get hybrid recommendations"""
        recommendations = []
        
        if book_title:
            similar_books = self.find_similar_books(book_title, top_n//2)
            recommendations.extend(similar_books)
        
        popular_books = self.get_popular_recommendations(top_n//2)
        recommendations.extend(popular_books)
        
        # Remove duplicates and limit to top_n
        seen_titles = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec['title'] not in seen_titles:
                unique_recommendations.append(rec)
                seen_titles.add(rec['title'])
                
        return unique_recommendations[:top_n]

def display_recommendations(recommendations, title="RECOMMENDATIONS"):
    """Display recommendations in formatted output"""
    print(f"\n{title}")
    print("=" * 70)
    
    if not recommendations:
        print("No recommendations available")
        return
        
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec['title']}")
        if 'author' in rec:
            print(f"     Author: {rec['author']}")
        if 'similarity_score' in rec:
            print(f"     Similarity: {rec['similarity_score']:.3f}")
        if 'ratings_count' in rec:
            print(f"     Ratings: {rec['ratings_count']}")
        print(f"     Type: {rec['type']}")
        print()

def main():
    """Main execution function"""
    print("=" * 60)
    print("BOOK RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    system = BookRecommendationSystem()
    
    # Load and prepare data
    if not system.load_data():
        return
    
    print("\nPreprocessing data...")
    processed_data = system.preprocess_data()
    
    print("\nBuilding recommendation models...")
    system.build_models(processed_data)
    
    print("\n" + "=" * 60)
    print("SYSTEM READY - Enter any book name to get recommendations!")
    print("=" * 60)
    
    while True:
        print("\nOPTIONS:")
        print("1. Search Books")
        print("2. Get Book Recommendations")
        print("3. Show Popular Books")
        print("4. Hybrid Recommendations")
        print("5. Exit")
        
        choice = input("\nChoose option (1-5): ").strip()
        
        if choice == '1':
            query = input("Enter book title or author to search: ").strip()
            if query:
                results = system.search_books(query)
                if not results.empty:
                    print(f"\nFound {len(results)} results:")
                    for idx, row in results.iterrows():
                        print(f"- '{row['Book-Title']}' by {row['Book-Author']}")
                else:
                    print("No books found matching your search.")
                    
        elif choice == '2':
            book_title = input("Enter any book title (full or partial): ").strip()
            if book_title:
                recommendations = system.find_similar_books(book_title)
                display_recommendations(recommendations, f"BOOKS SIMILAR TO '{book_title.upper()}'")
                
        elif choice == '3':
            recommendations = system.get_popular_recommendations()
            display_recommendations(recommendations, "MOST POPULAR BOOKS")
            
        elif choice == '4':
            book_title = input("Enter book title (or press Enter for popular books): ").strip()
            recommendations = system.hybrid_recommendations(book_title if book_title else None)
            title = "HYBRID RECOMMENDATIONS" 
            if book_title:
                title = f"HYBRID RECOMMENDATIONS BASED ON '{book_title.upper()}'"
            display_recommendations(recommendations, title)
            
        elif choice == '5':
            print("\nThank you for using the Book Recommendation System!")
            break
            
        else:
            print("Invalid option. Please choose 1-5.")

if __name__ == "__main__":
    main()