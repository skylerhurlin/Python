# This is a code I wrote to test inserting and updating random data in a SSMS database stored locally. 

import pyodbc

# Start with connecting to the database.

connect = None

try:
    connect = pyodbc.connect(Driver='SQL Server', host='localhost',
                             database='TestDB', user='skyef',
                             password='[censored password]'

except pyodbc.Error as e:
    print(f"Database connection unsuccessful: {e}. Please review and try again.")
    quit()

# Make a cursor to execute the query, starting with inserting data.

cursor = connect.cursor()

# Inserting the new data.

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
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# This executes the query and commits the new information to the database.

try:
    cursor.execute(sqlInsert, productID, productName, category, subcategory, price, discount, discounted, oosAmount, currentStock, supplierID)
    connect.commit()
    print("Product information added successfully.")

# Add an error message just in case.

except Exception as e:
    print(f"Error adding person information: {e}. Please review and try again.")
    quit()

# Moving on to updating information in a different table.

contactNumber = "555-555-5555"
contactName = "David Mays"
supplierID = 10

sqlUpdate = """
    UPDATE suppliers
     SET contactNumber = ?, 
         contactName = ?
     WHERE supplierID = ?
"""

# Executing the query and committing the new info.

try:

    cursor.execute(sqlUpdate, contactNumber, contactName, supplierID)
    connect.commit()
    print("Supplier information updated successfully.")

except Exception as e:

    print(f"Error updating supplier information: {e}. Please review and try again.")
    quit()

# Lastly, I want to select some information.

sqlSelect = """
    SELECT productName, category, subcategory
    FROM products
    WHERE supplierID = 10
    ORDER BY category, productName
"""

try:

    cursor.execute(sqlSelect)
    results = cursor.fetchall()
    for row in results:
        productName, category, subcategory = row
        print("{}, {}, {}".format(productName, category, subcategory))

except Exception as e:

    print(f"Error retrieving information: {e}. Please review and try again.")
    quit()

# When I'm done, I close the connection.

finally:

    cursor.close()
    connect.close()
