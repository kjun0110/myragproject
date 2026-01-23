import { Product } from "./types";

interface ProductCardProps {
  product: Product;
  onEdit?: () => void;
  onDelete?: () => void;
}

export default function ProductCard({ product, onEdit, onDelete }: ProductCardProps) {
  const getTypeLabel = (type: string) => {
    switch (type) {
      case "merchandise":
        return "상품";
      case "ticket":
        return "티켓";
      case "experience":
        return "체험";
      default:
        return type;
    }
  };

  return (
    <div className="product-card">
      <div className="product-header">
        <h4>{product.name}</h4>
        <span className={`product-type ${product.type}`}>
          {getTypeLabel(product.type)}
        </span>
      </div>
      <p className="product-description">{product.description}</p>
      <div className="product-info">
        <p>💰 가격: {product.price.toLocaleString()}원</p>
        <p>📦 재고: {product.stock}개</p>
      </div>
      <div className="product-actions">
        <button className="edit-button" onClick={onEdit}>수정</button>
        <button className="delete-button" onClick={onDelete}>삭제</button>
      </div>
      <style jsx>{`
        .product-card {
          border: 1px solid #e5e7eb;
          border-radius: 0.5rem;
          padding: 1.5rem;
          background: white;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .product-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .product-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .product-header h4 {
          margin: 0;
          font-size: 1.1rem;
        }

        .product-type {
          padding: 0.25rem 0.75rem;
          border-radius: 0.25rem;
          font-size: 0.75rem;
          font-weight: 600;
        }

        .product-type.merchandise {
          background: #dbeafe;
          color: #1e40af;
        }

        .product-type.ticket {
          background: #fef3c7;
          color: #92400e;
        }

        .product-type.experience {
          background: #fce7f3;
          color: #9f1239;
        }

        .product-description {
          color: #6b7280;
          margin-bottom: 1rem;
        }

        .product-info p {
          margin: 0.5rem 0;
          color: #6b7280;
        }

        .product-actions {
          display: flex;
          gap: 0.5rem;
          margin-top: 1rem;
        }

        .edit-button,
        .delete-button {
          flex: 1;
          padding: 0.5rem;
          border: 1px solid #d1d5db;
          border-radius: 0.25rem;
          background: white;
          cursor: pointer;
          transition: background 0.2s;
        }

        .edit-button {
          color: #3b82f6;
        }

        .edit-button:hover {
          background: #eff6ff;
        }

        .delete-button {
          color: #ef4444;
        }

        .delete-button:hover {
          background: #fef2f2;
        }
      `}</style>
    </div>
  );
}
