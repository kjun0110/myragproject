import { Product } from "./types";

interface PopularProductCardProps {
  product: Product;
}

export default function PopularProductCard({ product }: PopularProductCardProps) {
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
    <div className="popular-product-card">
      <div className="product-name-row">
        <h4>{product.name}</h4>
        <span className={`product-type-badge ${product.type}`}>
          {getTypeLabel(product.type)}
        </span>
      </div>
      <p className="product-price">{product.price.toLocaleString()}원</p>
      <p className="product-stock">재고: {product.stock}개</p>
      <style jsx>{`
        .popular-product-card {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 0.5rem;
          padding: 1rem;
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .popular-product-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .product-name-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .product-name-row h4 {
          margin: 0;
          font-size: 1rem;
          color: #333;
        }

        .product-type-badge {
          padding: 0.25rem 0.5rem;
          border-radius: 0.25rem;
          font-size: 0.7rem;
          font-weight: 600;
        }

        .product-type-badge.merchandise {
          background: #dbeafe;
          color: #1e40af;
        }

        .product-type-badge.ticket {
          background: #fef3c7;
          color: #92400e;
        }

        .product-type-badge.experience {
          background: #fce7f3;
          color: #9f1239;
        }

        .product-price {
          margin: 0.5rem 0;
          font-size: 1.1rem;
          font-weight: 600;
          color: #667eea;
        }

        .product-stock {
          margin: 0;
          font-size: 0.85rem;
          color: #6b7280;
        }
      `}</style>
    </div>
  );
}
