import React, { useState, useEffect } from 'react';
import { Modal, Button, Form } from 'react-bootstrap';
import './modal.css';

const categoriesData = {
  income: {
    salary: [],
    freelancing: ['Project 1', 'Project 2', 'Consulting'],
    investments: ['Stocks', 'Bonds', 'Real Estate'],
    bank: ['Savings', 'Checking', 'Wire Transfer'],
    youtube: ['Sponsorship', 'Membership', 'Adsense'],
    other: ['Gift', 'Bonus', 'Other Income'],
  },
  expense: {
    education: ['Tuition', 'Books', 'Supplies'],
    food: ['Restaurants', 'Snacks', 'Beverages'],
    groceries: ['Fruits', 'Vegetables', 'Dairy'],
    health: ['Medicine', 'Consultation', 'Fitness'],
    subscriptions: ['Streaming', 'Magazines', 'Memberships'],
    clothing: ['Shirts', 'Pants', 'Accessories'],
    travelling: ['Flights', 'Hotels', 'Transportation'],
    other: ['Miscellaneous', 'Uncategorized'],
  }
};

const UpdateModal = ({ show, handleClose, currentData, onSubmit, isEditable = true, toggleEditable }) => {
  const { type, category: currentCategory } = currentData;
  const [updatedData, setUpdatedData] = useState({
    title: currentData?.title || '',
    amount: currentData?.amount || '',
    category: '',
    subCategory: '',
    description: currentData?.description || '',
    date: currentData?.date?.split('T')[0] || '',
  });

  const categoryOptions = type === 'income' ? categoriesData.income : categoriesData.expense;

  useEffect(() => {
    const categoryParts = currentCategory ? currentCategory.split(' - ') : "Null";
    setUpdatedData((prevData) => ({
      ...prevData,
      category: categoryParts[0],
      subCategory: categoryParts[1] || '',
    }));
  }, [currentCategory]);

  // useEffect(() => {
  //   if (currentData) {
  //     const categoryParts = currentData.category ? currentData.category.split(' - ') : "Null";
  //     setUpdatedData({
  //       ...updatedData,
  //       title: currentData.title,
  //       amount: currentData.amount,
  //       category: categoryParts[0],
  //       subCategory: categoryParts[1] || '',
  //       description: currentData.description,
  //       date: currentData.date?.split('T')[0] || '',
  //     });
  //   }
  // }, [currentData]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setUpdatedData({ ...updatedData, [name]: value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const finalData = {
      ...updatedData,
      category: updatedData.subCategory ? `${updatedData.category} - ${updatedData.subCategory}` : updatedData.category,
    };
    onSubmit(finalData);
  };

  return (
    <Modal show={show} onHide={handleClose}>
      <Modal.Header closeButton>
        <Modal.Title>{isEditable ? `Update ${type === 'income' ? 'Income' : 'Expense'}` : `View ${type === 'income' ? 'Income' : 'Expense'}`}</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Form onSubmit={handleSubmit}>
          <Form.Group controlId="formTitle">
            <Form.Label>Title</Form.Label>
            <Form.Control
              type="text"
              name="title"
              value={updatedData.title}
              onChange={handleChange}
              required
              disabled={!isEditable}
            />
          </Form.Group>

          <Form.Group controlId="formAmount">
            <Form.Label>Amount</Form.Label>
            <Form.Control
              type="number"
              name="amount"
              value={updatedData.amount}
              onChange={handleChange}
              required
              disabled={!isEditable} 
            />
          </Form.Group>

          <Form.Group controlId="formCategory">
            <Form.Label>Category</Form.Label>
            <Form.Control
              as="select"
              name="category"
              value={updatedData.category}
              onChange={handleChange}
              required
              disabled={!isEditable} 
            >
              <option value="" disabled>Select Category</option>
              {Object.keys(categoryOptions).map((category, index) => (
                <option key={index} value={category}>
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </option>
              ))}
            </Form.Control>
          </Form.Group>

          {updatedData.category && categoryOptions[updatedData.category]?.length > 0 && (
            <Form.Group controlId="formSubCategory">
              <Form.Label>Subcategory</Form.Label>
              <Form.Control
                as="select"
                name="subCategory"
                value={updatedData.subCategory}
                onChange={handleChange}
                disabled={!isEditable}  // Disable if not editable
              >
                <option value="">Select Subcategory</option>
                {categoryOptions[updatedData.category].map((subCategory, index) => (
                  <option key={index} value={subCategory}>
                    {subCategory}
                  </option>
                ))}
              </Form.Control>
            </Form.Group>
          )}

          <Form.Group controlId="formDescription">
            <Form.Label>Description</Form.Label>
            <Form.Control
              as="textarea"
              name="description"
              value={Array.isArray(updatedData.description)
                ? updatedData.description
                  .map(
                    (desc) => `${desc.item} - Quantity: ${desc.quantity}, Price: ${desc.price}`
                  )
                  .join('\n') + `\nTax: Rs.${currentData.tax || 0}`
                : updatedData.description}
              onChange={handleChange}
              rows={4}
              required
              disabled={!isEditable}
            />
          </Form.Group>


          <Form.Group controlId="formDate">
            <Form.Label>Date</Form.Label>
            <Form.Control
              type="date"
              name="date"
              value={updatedData.date}
              onChange={handleChange}
              required
              disabled={!isEditable}
            />
          </Form.Group>

          <div className="modal-button-container">
            {!isEditable ? (
              <Button variant="primary" onClick={toggleEditable}>
                Edit
              </Button>
            ) : (
              <>
                <Button variant="secondary" onClick={handleClose}>
                  Cancel
                </Button>
                <Button className="Modal-submit" variant="primary" type="submit">
                  Update
                </Button>
              </>
            )}
          </div>
        </Form>
      </Modal.Body>
    </Modal>
  );
};

export default UpdateModal;
