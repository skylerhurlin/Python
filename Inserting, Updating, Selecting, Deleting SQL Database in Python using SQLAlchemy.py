from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Start with connecting to the database.

# Create an engine to connect to the SQL Server
engine = create_engine("mssql+pyodbc://<computerdrivername>/TestDB?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes")

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

try:
    print("Database connection successful.")

    # Start with inserting data.
    productName = "Tazo Tea: Earl Grey"
    category = "Beverages"
    subcategory = "Tea and Coffee"
    price = 5.50
    discount = 4.50
    discounted = 0
    oosAmount = 10
    currentStock = 25
    supplierID = 10

    sqlInsert = """
            INSERT INTO products (productName, category, subcategory, price, discount, discounted, oosAmount, currentStock, supplierID)
            VALUES (:productName, :category, :subcategory, :price, :discount, :discounted, :oosAmount, :currentStock, :supplierID)
        """
    # This executes the query and commits the new information to the database.

    session.execute(text(sqlInsert), {
        "productName": productName,
        "category": category,
        "subcategory": subcategory,
        "price": price,
        "discount": discount,
        "discounted": discounted,
        "oosAmount": oosAmount,
        "currentStock": currentStock,
        "supplierID": supplierID
    })

    session.commit()
    print("Product information added successfully.")

    # Now, updating information in a different table.

    contactNumber = "555-555-5555"
    contactName = "David Mays"
    supplierID = 10

    sqlUpdate = """
        UPDATE suppliers
        SET contactNumber = :contactNumber, 
            contactName = :contactName
        WHERE supplierID = :supplierID
    """

    # Executing the query and committing the new info.

    session.execute(text(sqlUpdate), {
        "contactNumber": contactNumber,
        "contactName": contactName,
        "supplierID": supplierID
    })

    session.commit()
    print("Supplier information updated successfully.")

    # Now, I want to select some information.

    sqlSelect = """
        SELECT productName, category, subcategory
        FROM products
        WHERE supplierID = 10
        ORDER BY category, productName
    """

    result = session.execute(text(sqlSelect), {"supplierID": supplierID})
    df = pd.DataFrame(result.fetchall(), columns=["productName", "category", "subcategory"])

    print(df)
    print(type(df))

    # Lastly, I will delete the product I inserted earlier.

    toDelete = "Tazo Tea: Earl Grey"

    sqlDelete = """
        DELETE FROM products
        WHERE productName = :productName
    """

    session.execute(text(sqlDelete), {"productName": toDelete})
    session.commit()
    print(f"Product deleted successfully.")

except Exception as e:

    print(f"Error: {e}. Please review and try again.")
    session.rollback()

# When I'm done, I close the connection.

finally:
    session.close()
    engine.dispose()
