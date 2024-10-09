import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import torch
import os, time

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, 
                 test_losses=None, test_accuracies=None):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')

    if test_losses is not None:
        plt.plot(epochs, test_losses, label='Test Loss', color='green')

    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')

    if test_accuracies is not None:
        plt.plot(epochs, test_accuracies, label='Test Accuracy', color='green')

    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    
def train_epoch(training_model, loader, criterion, optim, device):
    training_model.train()
    epoch_loss = 0.0
    all_labels = []
    all_predictions = []
    
    for images, labels in tqdm(loader):
      all_labels.extend(labels.numpy())

      optim.zero_grad()

      predictions = training_model(images.to(device))
      all_predictions.extend(torch.argmax(predictions, dim=1).cpu().numpy())

      loss = criterion(predictions, labels.to(device))
      
      loss.backward()
      optim.step()

      epoch_loss += loss.item()

    return epoch_loss / len(loader), accuracy_score(all_labels, all_predictions) * 100

def validation_epoch(val_model, loader, criterion, device):
    val_model.eval()
    epoch_loss = 0.0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
      for images, labels in loader:
        all_labels.extend(labels.numpy())  

        predictions = val_model(images.to(device))
        all_predictions.extend(torch.argmax(predictions, dim=1).cpu().numpy())

        loss = criterion(predictions, labels.to(device))

        epoch_loss += loss.item()

    return epoch_loss / len(loader), accuracy_score(all_labels, all_predictions) * 100 , all_labels, all_predictions

def other_epoch(val_model, other_loader, device):
  val_model.eval()
  epoch_acc = 0.0
  with torch.no_grad():
      for images, _ in other_loader:
          predictions = val_model(images.to(device))
          probabilities = torch.softmax(predictions,dim=1)
          max_prob, predicted_class = torch.max(probabilities, 1)
          epoch_acc += (max_prob<0.5).cpu().numpy().sum()/max_prob.shape

  return (epoch_acc / len(other_loader))

def train_model(model, train_loader, val_loader, criterion, optim, device, number_epochs=50, patience=5):
  train_loss_history = []
  val_loss_history = []
  val_acc_history = []
  train_acc_history = []
  best_acc = 0.0
  retry=0
  best_epoch = 0

  for epoch in range(number_epochs):
      start_time = time.time()

      train_loss, train_acc = train_epoch(model, train_loader, criterion, optim, device)
      train_loss_history.append(train_loss)
      train_acc_history.append(train_acc)
      print("Training epoch {} | Loss {:.6f} | Accuracy {:.2f}% | Time {:.2f} seconds"
            .format(epoch + 1, train_loss, train_acc, time.time() - start_time))

      start_time = time.time()
      val_loss, val_acc, _, _ = validation_epoch(model, val_loader, criterion, device)
      val_loss_history.append(val_loss)
      val_acc_history.append(val_acc)
      print("Validation epoch {} | Loss {:.6f} | Accuracy {:.2f}% | Time {:.2f} seconds"
            .format(epoch + 1, val_loss, val_acc, time.time() - start_time))      
            
      if val_acc > best_acc:
         best_acc = val_acc
         torch.save(model.state_dict(), "best_model.pt")
         retry=0
         best_epoch = epoch + 1
      else:
         retry+=1
         print(f'Retry:{retry}')
         if retry>patience:
            print("Early stopping")
            break
  print(f'Best Epoch: {best_epoch}')
  model.load_state_dict(torch.load("best_model.pt"))
  val_loss, val_acc, labels, prediction = validation_epoch(model, val_loader, criterion, device)
  print("Validation - Best Model | Loss {:.6f} | Accuracy {:.2f}%"
            .format(val_loss, val_acc))
  
  print(train_loader.dataset.dataset.class_to_idx)
  print(classification_report(prediction,labels))
  ConfusionMatrixDisplay(confusion_matrix(prediction,labels)).plot()  

  plot_metrics(train_loss_history,val_loss_history,train_acc_history,val_acc_history)

def train_model_wtest(model, train_loader, val_loader, test_loader, criterion, optim, device, number_epochs=50, patience=5):
  train_loss_history = []
  val_loss_history = []
  val_acc_history = []
  train_acc_history = []
  test_loss_history = []
  test_acc_history = []
  best_acc = 0.0
  retry=0
  best_epoch = 0

  for epoch in range(number_epochs):
      start_time = time.time()

      train_loss, train_acc = train_epoch(model, train_loader, criterion, optim, device)
      train_loss_history.append(train_loss)
      train_acc_history.append(train_acc)
      print("Training epoch {} | Loss {:.6f} | Accuracy {:.2f}% | Time {:.2f} seconds"
            .format(epoch + 1, train_loss, train_acc, time.time() - start_time))

      start_time = time.time()
      val_loss, val_acc, _, _ = validation_epoch(model, val_loader, criterion, device)
      val_loss_history.append(val_loss)
      val_acc_history.append(val_acc)
      print("Validation epoch {} | Loss {:.6f} | Accuracy {:.2f}% | Time {:.2f} seconds"
            .format(epoch + 1, val_loss, val_acc, time.time() - start_time))

      start_time = time.time()
      test_loss, test_acc, _, _= validation_epoch(model, test_loader, criterion, device)
      test_loss_history.append(test_loss)
      test_acc_history.append(test_acc)
      print("Test epoch {} | Loss {:.6f} | Accuracy {:.2f}% | Time {:.2f} seconds"
            .format(epoch + 1, test_loss, test_acc, time.time() - start_time))
           
      if val_acc > best_acc:
         best_acc = val_acc
         torch.save(model.state_dict(), "best_model.pt")
         retry=0
         best_epoch = epoch + 1
      else:
         retry+=1
         print(f'Retry:{retry}')
         if retry>patience:
            print("Early stopping")
            break
  print(f'Best Epoch: {best_epoch}')
  model.load_state_dict(torch.load("best_model.pt"))
  val_loss, val_acc, labels, prediction = validation_epoch(model, val_loader, criterion, device)
  print("Validation - Best Model | Loss {:.6f} | Accuracy {:.2f}%"
            .format(val_loss, val_acc))
  test_loss, test_acc, _, _ = validation_epoch(model, test_loader, criterion, device)
  print("Test - Best Model | Loss {:.6f} | Accuracy {:.2f}%"
            .format(test_loss, test_acc))
  
  print(train_loader.dataset.dataset.class_to_idx)
  print(classification_report(prediction,labels))
  ConfusionMatrixDisplay(confusion_matrix(prediction,labels)).plot()
  
  plot_metrics(train_loss_history,val_loss_history,train_acc_history,val_acc_history,test_loss_history,test_acc_history)

def train_model_wtest_wother(model, train_loader, val_loader, test_loader, other_loader, criterion, optim, device, number_epochs=50, patience=5):
  train_loss_history = []
  val_loss_history = []
  val_acc_history = []
  train_acc_history = []
  test_loss_history = []
  test_acc_history = []
  best_acc = 0.0
  retry=0
  best_epoch = 0

  for epoch in range(number_epochs):
      start_time = time.time()

      train_loss, train_acc = train_epoch(model, train_loader, criterion, optim, device)
      train_loss_history.append(train_loss)
      train_acc_history.append(train_acc)
      print("Training epoch {} | Loss {:.6f} | Accuracy {:.2f}% | Time {:.2f} seconds"
            .format(epoch + 1, train_loss, train_acc, time.time() - start_time))

      start_time = time.time()
      val_loss, val_acc, _, _ = validation_epoch(model, val_loader, criterion, device)
      val_loss_history.append(val_loss)
      val_acc_history.append(val_acc)
      print("Validation epoch {} | Loss {:.6f} | Accuracy {:.2f}% | Time {:.2f} seconds"
            .format(epoch + 1, val_loss, val_acc, time.time() - start_time))

      start_time = time.time()
      test_loss, test_acc, _, _ = validation_epoch(model, test_loader, criterion, device)
      test_loss_history.append(test_loss)
      test_acc_history.append(test_acc)
      print("Test epoch {} | Loss {:.6f} | Accuracy {:.2f}% | Time {:.2f} seconds"
            .format(epoch + 1, test_loss, test_acc, time.time() - start_time))

      start_time = time.time()
      other_acc = other_epoch(model, other_loader, device)
      print("Other epoch {} | Accuracy {:.2f}% | Time {:.2f} seconds"
            .format(epoch + 1, other_acc[0]*100 , time.time() - start_time))
            
      if val_acc > best_acc:
         best_acc = val_acc
         torch.save(model.state_dict(), "best_model.pt")
         retry=0
         best_epoch = epoch + 1
      else:
         retry+=1
         print(f'Retry:{retry}')
         if retry>patience:
            print("Early stopping")
            break
  print(f'Best Epoch: {best_epoch}')
  model.load_state_dict(torch.load("best_model.pt"))
  val_loss, val_acc, labels, prediction = validation_epoch(model, val_loader, criterion, device)
  print("Validation - Best Model | Loss {:.6f} | Accuracy {:.2f}%"
            .format(val_loss, val_acc))
  test_loss, test_acc, _, _ = validation_epoch(model, test_loader, criterion, device)
  print("Test - Best Model | Loss {:.6f} | Accuracy {:.2f}%"
            .format(test_loss, test_acc))
  other_acc = other_epoch(model, other_loader, device)
  print("Other - Best Model | Accuracy {:.2f}%"
        .format(other_acc[0]*100))
  
  print(train_loader.dataset.dataset.class_to_idx)
  print(classification_report(prediction,labels))
  ConfusionMatrixDisplay(confusion_matrix(prediction,labels)).plot()
  
  plot_metrics(train_loss_history,val_loss_history,train_acc_history,val_acc_history,test_loss_history,test_acc_history)
